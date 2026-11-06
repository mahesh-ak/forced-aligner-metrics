"""GPU-accelerated numerical kernels used by :mod:`forced_aligner_metrics`.

All heavy pairing / clustering metrics that the original implementation
executed sequentially with ``fastdtw``, ``sklearn`` and CPU numpy are
reimplemented here with torch so that they can run as batched, vectorized
operations on a CUDA device:

* :func:`batched_dtw_cosine_sims` - exact dynamic time warping with cosine
  distance, evaluated for thousands of sequence pairs at once via an
  anti-diagonal sweep through the padded batch of cost matrices.
* :func:`kmeans_fit_predict` - k-means (k-means++ init + Lloyd iterations)
  with GPU distance computations.
* :func:`normalized_mutual_info` - normalized mutual information computed
  from a GPU contingency matrix.

Every function also works on CPU torch tensors, which keeps the code
testable without a CUDA device.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

_INF = 1e9  # large positive sentinel for DTW cost masking


def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
    norms = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return x / torch.clamp(norms, min=1e-8)


def _pool_rows(x: torch.Tensor, k: int = 2) -> torch.Tensor:
    n = x.shape[0] // k
    if n == 0:
        return x
    return x[: n * k].reshape(n, k, -1).mean(dim=1)


def _dtw_chunk(xs: List[torch.Tensor], ys: List[torch.Tensor]) -> List[Tuple[float, int]]:
    """Exact DTW with cosine distance for a chunk of sequence pairs.

    Returns ``(total_distance, path_length)`` per pair. Sequences are
    row-normalized and pooled (mean over ``k`` consecutive frames) exactly
    like ``eval_aligns.pool_emb`` + ``eval_aligns.dtw_sim``.
    """
    B = len(xs)

    normed = []
    lengths_x = []
    for x in xs:
        # rows are normalized before pooling (as in dtw_sim) and the pooled
        # rows are re-normalized because cosine distance is computed between
        # pooled rows, which are not unit-length themselves.
        x = _normalize_rows(x.float())
        x = _pool_rows(x)
        x = _normalize_rows(x)
        lengths_x.append(x.shape[0])
        normed.append(x)

    lengths_y = []
    for i, y in enumerate(ys):
        y = _normalize_rows(y.float())
        y = _pool_rows(y)
        y = _normalize_rows(y)
        lengths_y.append(y.shape[0])
        ys[i] = y

    Nx = max(lengths_x)
    Ny = max(lengths_y)
    d = normed[0].shape[1]
    dev = normed[0].device

    X = torch.zeros(B, Nx, d, device=dev, dtype=normed[0].dtype)
    Y = torch.zeros(B, Ny, d, device=dev, dtype=normed[0].dtype)
    for b, (x, y) in enumerate(zip(normed, ys)):
        X[b, : x.shape[0]] = x
        Y[b, : y.shape[0]] = y

    lx = torch.tensor(lengths_x, device=dev)
    ly = torch.tensor(lengths_y, device=dev)

    # cosine distance cost matrices, padded cells masked with +INF
    cost = 1.0 - torch.bmm(X, Y.transpose(1, 2))  # (B, Nx, Ny)
    valid = (torch.arange(Nx, device=dev).view(1, Nx, 1) < lx.view(B, 1, 1)) & (
        torch.arange(Ny, device=dev).view(1, 1, Ny) < ly.view(B, 1, 1)
    )
    cost = torch.where(valid, cost, torch.full_like(cost, _INF))

    D = torch.full((B, Nx, Ny), _INF, device=dev, dtype=cost.dtype)
    S = torch.zeros((B, Nx, Ny), device=dev, dtype=torch.int64)

    for k in range(Nx + Ny - 1):
        i_lo = max(0, k - (Ny - 1))
        i_hi = min(k, Nx - 1)
        i = torch.arange(i_lo, i_hi + 1, device=dev)
        j = k - i

        if k == 0:
            D[:, 0, 0] = cost[:, 0, 0]
            S[:, 0, 0] = 1
            continue

        # predecessor candidates; gathers are index-clamped and logically
        # masked, out-of-sequence cells stay +INF in D so they never win.
        up_ok = (i >= 1).view(1, -1)
        left_ok = (j >= 1).view(1, -1)
        diag_ok = up_ok & left_ok

        iu = torch.clamp(i - 1, min=0)
        jl = torch.clamp(j - 1, min=0)

        c1 = torch.where(up_ok, D[:, iu, j], torch.full_like(D[:, iu, j], _INF))
        c2 = torch.where(left_ok, D[:, i, jl], torch.full_like(D[:, i, jl], _INF))
        c3 = torch.where(diag_ok, D[:, iu, jl], torch.full_like(D[:, iu, jl], _INF))
        s1 = S[:, iu, j]
        s2 = S[:, i, jl]
        s3 = S[:, iu, jl]

        stacked = torch.stack([c1, c2, c3], dim=0)  # (3, B, L)
        best = stacked.min(dim=0).values
        arg = stacked.argmin(dim=0)
        chosen_steps = torch.stack([s1, s2, s3], dim=0).gather(0, arg.unsqueeze(0)).squeeze(0)

        newv = best + cost[:, i, j]
        cell_valid = valid[:, i, j]
        newv = torch.where(cell_valid, newv, torch.full_like(newv, _INF))
        D[:, i, j] = newv
        S[:, i, j] = torch.where(cell_valid, chosen_steps + 1, torch.zeros_like(chosen_steps))

    results = []
    for b in range(B):
        ex = lengths_x[b] - 1
        ey = lengths_y[b] - 1
        dist = D[b, ex, ey].item()
        plen = S[b, ex, ey].item()
        results.append((dist, plen))
    return results


def batched_dtw_cosine_sims(
    pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
    device: str = "cpu",
    chunk_size: int = 256,
) -> List[Optional[float]]:
    """Compute ``dtw_sim`` for many (x, y) sequence pairs in batched torch.

    Mirrors ``eval_aligns.dtw_sim``: rows are normalized, frames are pooled
    with ``k=2``, the distance is fastdtw-style cosine DTW and the score is
    ``1 - total_distance / path_length``.

    Parameter order in the returned list matches ``pairs``; pairs that
    would raise/return None in the sequential implementation (either
    sequence shorter than 2 frames) are returned as ``None``.
    """
    keep = [i for i, (x, y) in enumerate(pairs) if len(x) >= 2 and len(y) >= 2]
    out: List[Optional[float]] = [None] * len(pairs)

    if not keep:
        return out

    def _to_torch(x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.detach().to(device=device, dtype=torch.float32)
        return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=device)

    xs = [_to_torch(pairs[i][0]) for i in keep]
    ys = [_to_torch(pairs[i][1]) for i in keep]

    for start in range(0, len(keep), chunk_size):
        chunk_x = xs[start : start + chunk_size]
        chunk_y = ys[start : start + chunk_size]
        results = _dtw_chunk(chunk_x, chunk_y)
        for offset, (dist, plen) in enumerate(results):
            idx = keep[start + offset]
            out[idx] = 1.0 - dist / plen

    return out


def kmeans_fit_predict(
    X: torch.Tensor,
    n_clusters: int,
    seed: int = 0,
    max_iter: int = 300,
) -> torch.Tensor:
    """Deterministic GPU k-means (k-means++ init, Lloyd iterations).

    A drop-in replacement for ``sklearn.cluster.MiniBatchKMeans.fit_predict``
    on the PCMI feature matrix: with a fixed ``seed`` the result is fully
    reproducible, which also stabilizes the reported PCMI score.

    Parameter
    ---------
    X:
        ``(n_samples, n_features)`` float tensor (on any torch device).
    n_clusters:
        Number of clusters.
    seed:
        Seed used for the k-means++ initialization.
    max_iter:
        Maximum number of Lloyd iterations (usually converges in <50).

    Returns
    -------
    LongTensor of cluster assignments, shape ``(n_samples,)`` on X's device.
    """
    n, d = X.shape
    device = X.device
    X = X.float()
    g = torch.Generator(device="cpu")  # generator choice kept device-agnostic
    g.manual_seed(seed)

    if n_clusters >= n:
        # degenerate case: every point becomes its own cluster id by order
        return torch.arange(n, device=device)

    # ---- k-means++ initialization ----
    first = int(torch.randint(n, (1,), generator=g).item())
    centers = [X[first].clone()]
    closest_d2 = torch.cdist(X, centers[0].view(1, -1)).squeeze(1) ** 2

    for _ in range(1, n_clusters):
        probs = torch.clamp(closest_d2, min=0.0)
        total = probs.sum()
        if total.item() <= 0:
            pick = int(torch.randint(n, (1,), generator=g).item())
        else:
            # deterministic inverse-CDF draw from the D^2 distribution
            r = float(torch.rand(1, generator=g).item()) * float(total.item())
            csum = torch.cumsum(probs, dim=0)
            pick = int(torch.searchsorted(csum, torch.tensor(r, device=csum.device)).item())
            pick = min(pick, n - 1)
        centers.append(X[pick].clone())
        d2 = torch.cdist(X, X[pick].view(1, -1)).squeeze(1) ** 2
        closest_d2 = torch.minimum(closest_d2, d2)

    C = torch.stack(centers)  # (k, d)

    # ---- Lloyd iterations ----
    labels = torch.full((n,), -1, dtype=torch.long, device=device)
    for _ in range(max_iter):
        dist = torch.cdist(X, C)  # (n, k)
        new_labels = dist.argmin(dim=1)
        if torch.equal(new_labels, labels):
            break
        labels = new_labels
        # recompute centers; empty clusters keep their previous center
        sums = torch.zeros_like(C)
        counts = torch.zeros(n_clusters, device=device, dtype=torch.float32)
        sums.index_add_(0, labels, X)
        counts.index_add_(0, labels, torch.ones(n, device=device))
        moved = counts > 0
        C = torch.where(moved.view(-1, 1), sums / counts.clamp(min=1).view(-1, 1), C)

    return labels


def normalized_mutual_info(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Normalized mutual information (arithmetic normalization) on torch.

    Matches ``sklearn.metrics.normalized_mutual_info_score`` with
    ``average_method="arithmetic"`` (the scikit-learn default).
    """
    import math

    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    device = "cpu"
    mapping_true = {v: i for i, v in enumerate(np.unique(labels_true))}
    mapping_pred = {v: i for i, v in enumerate(np.unique(labels_pred))}
    a = torch.as_tensor([mapping_true[v] for v in labels_true], dtype=torch.long, device=device)
    b = torch.as_tensor([mapping_pred[v] for v in labels_pred], dtype=torch.long, device=device)

    ka = len(mapping_true)
    kb = len(mapping_pred)
    contingency = torch.bincount(a * kb + b, minlength=ka * kb).double()
    contingency = contingency.reshape(ka, kb)
    n = float(contingency.sum().item())

    pij = contingency / n
    pi = pij.sum(dim=1)
    pj = pij.sum(dim=0)

    # entropies and mutual information in float64 for numerical stability
    def _h(p):
        p = p[p > 0]
        return float((-p * torch.log(p)).sum().item())

    mi = float(
        (
            pij[(pij > 0)]
            * torch.log(pij[(pij > 0)] / ((pi.view(-1, 1) * pj.view(1, -1))[(pij > 0)]))
        )
        .sum()
        .item()
    )
    ha = _h(pi)
    hb = _h(pj)
    denom = (ha + hb) / 2.0
    if denom == 0.0:
        return 1.0 if mi == 0.0 else 0.0
    return mi / denom
