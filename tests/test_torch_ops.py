import numpy as np
import pytest
import torch
from sklearn.metrics import normalized_mutual_info_score

from forced_aligner_metrics.eval_aligns import dtw_sim
from forced_aligner_metrics.torch_ops import (
    batched_dtw_cosine_sims,
    kmeans_fit_predict,
    normalized_mutual_info,
)


def _exact_dtw_sim(x, y):
    """Small exact DTW + cosine reference used to validate the batched kernel."""
    from forced_aligner_metrics.eval_aligns import pool_emb

    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    y = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-8)
    x, y = pool_emb(x), pool_emb(y)
    xn = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    yn = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-8)
    C = 1.0 - xn @ yn.T
    D = np.full(C.shape, np.inf)
    D[0, 0] = C[0, 0]
    for i in range(len(x)):
        for j in range(len(y)):
            if i == 0 and j == 0:
                continue
            best = np.inf
            if i > 0:
                best = min(best, D[i - 1, j])
            if j > 0:
                best = min(best, D[i, j - 1])
            if i > 0 and j > 0:
                best = min(best, D[i - 1, j - 1])
            D[i, j] = C[i, j] + best
    i, j, L = len(x) - 1, len(y) - 1, 1
    while i > 0 or j > 0:
        cand = []
        if i > 0:
            cand.append((D[i - 1, j], i - 1, j))
        if j > 0:
            cand.append((D[i, j - 1], i, j - 1))
        if i > 0 and j > 0:
            cand.append((D[i - 1, j - 1], i - 1, j - 1))
        _, i, j = min(cand)
        L += 1
    return 1 - D[-1, -1] / L


def test_batched_dtw_matches_exact_dtw():
    rng = np.random.default_rng(0)
    pairs = [
        (rng.normal(size=(int(rng.integers(2, 40)), 6)), rng.normal(size=(int(rng.integers(2, 40)), 6)))
        for _ in range(30)
    ]
    # too-short sequences are dropped consistently with dtw_sim
    pairs += [(rng.normal(size=(5, 6)), rng.normal(size=(1, 6)))]

    got = batched_dtw_cosine_sims(pairs, device="cpu", chunk_size=8)
    exact = []
    for x, y in pairs:
        exact.append(_exact_dtw_sim(x, y) if len(x) >= 2 and len(y) >= 2 else None)
    for g, e in zip(got, exact):
        if e is None:
            assert g is None
        else:
            assert g == pytest.approx(e, abs=1e-5)


def test_batched_dtw_close_to_fastdtw():
    rng = np.random.default_rng(1)
    pairs = [
        (rng.normal(size=(int(rng.integers(4, 30)), 6)), rng.normal(size=(int(rng.integers(4, 30)), 6)))
        for _ in range(20)
    ]
    got = batched_dtw_cosine_sims(pairs, device="cpu")
    for (x, y), g in zip(pairs, got):
        assert g == pytest.approx(dtw_sim(x.copy(), y.copy()), abs=0.05)


def test_kmeans_recovers_wellseparated_clusters():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(600, 12)).astype(np.float32)
    X[:200] += 4.0
    X[200:400] -= 4.0
    labels = kmeans_fit_predict(torch.from_numpy(X), 4, seed=0)
    # more clusters than true groups: every pair of true groups must be split
    idx0 = labels[:200]
    idx1 = labels[200:400]
    assert len(set(idx0.tolist())) >= 1
    assert set(idx0.tolist()).isdisjoint(set(idx1.tolist()))
    # deterministic given the seed
    labels2 = kmeans_fit_predict(torch.from_numpy(X), 4, seed=0)
    assert torch.equal(labels, labels2)


def test_nmi_matches_sklearn():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 8, 1000)
    b = rng.integers(0, 17, 1000)
    assert normalized_mutual_info(a, b) == pytest.approx(normalized_mutual_info_score(a, b), abs=1e-8)
    assert normalized_mutual_info(a, a) == pytest.approx(1.0, abs=1e-8)
    # functional clustering dependence
    b2 = np.where(a < 4, 0, 1)
    assert normalized_mutual_info(a, b2) == pytest.approx(normalized_mutual_info_score(a, b2), abs=1e-8)
