import warnings
import numpy as np
import torch
from difflib import SequenceMatcher
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, Iterator, Dict, Any, List
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import normalized_mutual_info_score
from fastdtw import fastdtw
from scipy.spatial.distance import cosine
from collections import Counter
from scipy.stats import entropy

from forced_aligner_metrics.torch_ops import (
    batched_dtw_cosine_sims,
    kmeans_fit_predict,
    normalized_mutual_info,
)



class W2V2Embedding:

    def __init__(self, model_str, layer_range = (12, 16), device = "cuda", as_torch = False):
        self.device = device
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_str)
        self.model = Wav2Vec2Model.from_pretrained(model_str).to(self.device)
        self.sampling_rate = self.processor.sampling_rate
        self.layer_range = layer_range
        self.model.eval()
        # Freeze parameters so the forward pass builds no autograd graph.
        # With frozen params and non-grad inputs this is equivalent to
        # running under no_grad, without importing a specific framework.
        self.model.requires_grad_(False)
        # When True, embeddings stay on the compute device as torch tensors
        # instead of being copied to CPU numpy arrays after every batch.
        self.as_torch = as_torch

    def _feat_extract_output_lengths(self, input_lengths):
        """Compute exact wav2vec2 frame counts in numpy.

        Mirrors ``Wav2Vec2Model._get_feat_extract_output_lengths`` using the
        convolutional feature encoder's kernel sizes and strides, avoiding a
        direct dependency on the underlying tensor framework.
        """
        lengths = np.asarray(input_lengths)
        for kernel_size, stride in zip(self.model.config.conv_kernel, self.model.config.conv_stride):
            lengths = (lengths - kernel_size) // stride + 1
        return lengths

    def extract_embeddings_batch(self, audios):
        results = []

        inputs = self.processor(
            audios,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True
        )

        input_values = inputs.input_values.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        outputs = self.model(input_values, attention_mask=attention_mask, output_hidden_states=True)

        lo, hi = self.layer_range

        if self.as_torch:
            # keep everything on the compute device: no per-layer CPU roundtrip
            hidden = torch.stack(list(outputs.hidden_states[lo:hi])).mean(0)

            # sample lengths
            input_lengths = attention_mask.sum(-1).cpu().numpy()

            # exact wav2vec2 frame lengths
            frame_lengths = self._feat_extract_output_lengths(input_lengths)

            for i in range(len(audios)):

                num_samples = int(input_lengths[i])
                duration = num_samples / self.sampling_rate
                num_frames = int(frame_lengths[i])

                emb = hidden[i, :num_frames].contiguous()

                # frame timestamps
                frame_times = np.linspace(0,duration,num_frames,endpoint=False)

                results.append((emb, frame_times))

            return results

        hidden = np.stack(
            [h.cpu().numpy() for h in outputs.hidden_states[lo:hi]]
        ).mean(0)

        # sample lengths
        input_lengths = attention_mask.sum(-1).cpu().numpy()

        # exact wav2vec2 frame lengths
        frame_lengths = self._feat_extract_output_lengths(input_lengths)

        for i in range(len(audios)):

            num_samples = int(input_lengths[i])
            duration = num_samples / self.sampling_rate
            num_frames = int(frame_lengths[i])

            emb = hidden[i, :num_frames]

            # frame timestamps
            frame_times = np.linspace(0,duration,num_frames,endpoint=False)

            results.append((emb, frame_times))

        return results


def normalize_intervals(intervals: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Tuple[float, float, str]]]:
    normalized = {}
    for tier_name, tier_intervals in (intervals or {}).items():
        normalized_tier = []
        for entry in tier_intervals:
            if isinstance(entry, dict):
                start = float(entry.get("start", entry.get("start_time", 0.0)))
                end = float(entry.get("end", entry.get("end_time", start)))
                label = str(entry.get("text", "")).strip()
            else:
                continue

            if not label:
                label = "<SIL>"
            normalized_tier.append((start, end, label))
        normalized[tier_name] = normalized_tier
    return normalized


def compute_aas(gold: Dict[str, List[Dict[str, Any]]], pred: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Absolute boundary deviation (AAS) between two alignments, in ms.

    Parameters
    ----------
    gold, pred:
        Interval dicts in the same format as the other metrics
        (``{"words": [{"text", "start", "end"}, ...], "phones": [...]}``).
        Each tier present in both is aligned with a ``SequenceMatcher`` over
        the label sequence: equal pairs contribute ``(|dstart| + |dend|) / 2``
        to the error, non-equal (insertion/deletion/substitution) gold
        intervals contribute their full duration.

    Returns
    -------
    ``{"word_aas_ms": ..., "phone_aas_ms": ...}`` - the mean error in
    milliseconds per tier (``None`` for a tier missing on either side or
    with no errors).

    Silence spans are treated as sequence elements of their own (label
    ``"<SIL>"``): empty-label intervals become ``"<SIL>"`` entries, and a gap
    between two adjacent intervals is assumed to be a silence span as well.
    Silence is then scored like any other interval - equal silence pairs
    contribute their boundary deviation, unmatched silence contributes its
    full duration.
    """

    def norm_tier(intervals, tier):
        SIL = "<SIL>"
        raw: List[Tuple[float, float, str]] = []
        prev_end = None
        for entry in (intervals or {}).get(tier, []):
            if not isinstance(entry, dict):
                continue
            s = float(entry.get("start", entry.get("start_time", 0.0)))
            e = float(entry.get("end", entry.get("end_time", s)))
            lab = str(entry.get("text", "")).strip() or SIL
            if prev_end is not None and s > prev_end + 1e-9:
                # adjacent interval begins later: assume silence in the gap
                raw.append((prev_end, s, SIL))
            raw.append((s, e, lab))
            prev_end = e if prev_end is None else max(prev_end, e)
        # merge contiguous silence spans into single "<SIL>" entries
        out: List[Tuple[float, float, str]] = []
        for s, e, lab in raw:
            if out and lab == SIL and out[-1][2] == SIL and s <= out[-1][1] + 1e-9:
                out[-1] = (out[-1][0], max(out[-1][1], e), SIL)
            else:
                out.append((s, e, lab))
        return out

    def align_and_score(g, p):
        sm = SequenceMatcher(a=[x[2] for x in g], b=[x[2] for x in p])
        errs = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for gx, px in zip(g[i1:i2], p[j1:j2]):
                    errs.append((abs(gx[0] - px[0]) + abs(gx[1] - px[1])) / 2)
            else:
                for x in g[i1:i2]:
                    errs.append(x[1] - x[0])
        if not errs:
            return None
        return 1000.0 * float(np.mean(errs))

    out: Dict[str, Any] = {}
    for tier, key in (("words", "word_aas_ms"), ("phones", "phone_aas_ms")):
        g = norm_tier(gold, tier)
        p = norm_tier(pred, tier)
        if g and p:
            aas = align_and_score(g, p)
            out[key] = round(aas, 3) if aas is not None else None
        else:
            out[key] = None
    return out


def assign_labels_to_frames(frame_times, intervals):
    labels = []
    j = 0
    for t in frame_times:
        while j < len(intervals) - 1 and t > intervals[j][1]:
            j += 1
        labels.append(intervals[j][2])
    return labels


def pool_emb(x, k=2):
    n=len(x)//k
    if n==0: return x
    return x[:n*k].reshape(n,k,-1).mean(1)

def crop_frames(emb, frame_times, start, end):
    idx=(frame_times>=start)&(frame_times<=end)
    if isinstance(emb, torch.Tensor):
        return emb[torch.as_tensor(idx, device=emb.device)]
    return emb[idx]

def dtw_sim(x,y):
    if len(x)<2 or len(y)<2: return None
    x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)
    y=y/np.maximum(np.linalg.norm(y,axis=1,keepdims=True),1e-8)
    x,y=pool_emb(x),pool_emb(y)
    distance, path =fastdtw(x,y,dist=cosine)
    return 1 - distance/len(path)

@dataclass
class ForcedAlignEvalConfig:
    """Configuration for the import-based evaluation API.

    Parameters
    ----------
    model:
        Hugging Face model name used for Wav2Vec2 embeddings.
    device:
        Device string passed to the underlying model (e.g. ``"cpu"`` or
        ``"cuda"``). The choice of compute backend is left to the
        ``transformers`` library and its installed framework.
    layer_range:
        Hidden layer range to average when extracting embeddings.
    batch_size:
        Number of samples processed per embedding batch.
    samples_phon_eval:
        Number of phone-level evaluation samples to collect.
    max_frames:
        Maximum frame count used for clustering-based PCMI evaluation.
    n_clusters:
        Number of clusters used by the PCMI clustering step.
    samples_word_eval:
        Number of word-level samples to collect for WACS.
    max_words:
        Maximum number of vocabulary items used for WACS.
    min_occ:
        Minimum occurrence count required for a word to enter the WACS vocabulary.
    max_pairs:
        Maximum number of positive/negative DTW pairs per word.
    backend:
        Which kernels drive DTW (WACS), clustering and NMI (PCMI).
        ``"auto"`` (default) uses the batched torch/GPU kernels whenever
        ``device`` is a CUDA device and falls back to the original
        numpy/fastdtw/sklearn implementation otherwise; ``"torch"``
        forces the torch kernels (also usable on CPU tensors);
        ``"legacy"`` always uses the original numpy implementation.
    kmeans_seed:
        Seed for the GPU k-means initialization (torch backend only).
        The legacy MiniBatchKMeans path remains unseeded/global-RNG driven.
    dtw_chunk_size:
        Number of word pairs evaluated per batched DTW kernel
        (torch backend only).
    """

    model : str = "facebook/mms-300m"
    device : str = "cuda"
    layer_range: Tuple[int, int] = (15, 16)
    batch_size : int = 2

    samples_phon_eval : int = 50
    max_frames : int = 10_000
    n_clusters : int = 50

    samples_word_eval : int = 200
    max_words: int = 200
    min_occ : int = 3
    max_pairs : int = 10

    backend : str = "auto"
    kmeans_seed : int = 0
    dtw_chunk_size : int = 256


class ForcedAlignEval:
    """Main import-based evaluator for PCMI and WACS metrics.

    Parameters
    ----------
    config:
        A ForcedAlignEvalConfig instance defining the evaluation behavior.
    """

    def __init__(self, config=ForcedAlignEvalConfig()):
        self.config=config
        self.backend = self._resolve_backend(config)
        self.emb_model = W2V2Embedding(
            config.model, config.layer_range, config.device,
            as_torch=(self.backend == "torch"),
        )

    @staticmethod
    def _resolve_backend(config):
        backend = getattr(config, "backend", "auto")
        if backend == "auto":
            backend = "torch" if config.device.startswith("cuda") else "legacy"
        if backend not in ("torch", "legacy"):
            raise ValueError(f"Unknown backend: {backend!r} (expected 'auto', 'torch' or 'legacy')")
        return backend

    def collect_embeddings(self, audios: Iterator[np.ndarray], intervals_iter: Iterator[Dict[str, List[Dict[str, Any]]]]) -> dict:

        all_frame_embs = []
        all_frame_labels = []
        word2embs = defaultdict(list)

        processed = 0
        sample_size = max(
            self.config.samples_word_eval,
            self.config.samples_phon_eval,
        )
        batch_size = self.config.batch_size

        while processed < sample_size:
            batch_audios = []
            batch_intervals = []

            for _ in range(min(batch_size, sample_size - processed)):
                try:
                    audio = next(audios)
                    intervals = next(intervals_iter)
                except StopIteration:
                    break

                batch_audios.append(audio)
                batch_intervals.append(intervals)

            if not batch_audios:
                break

            batch_results = self.emb_model.extract_embeddings_batch(batch_audios)

            for batch_idx, (emb, frame_times) in enumerate(batch_results):
                normalized_intervals = normalize_intervals(batch_intervals[batch_idx])

                if processed < self.config.samples_phon_eval and "phones" in normalized_intervals:
                    labels = assign_labels_to_frames(frame_times, normalized_intervals["phones"])
                    all_frame_embs.append(emb)
                    all_frame_labels.extend(labels)

                if processed < self.config.samples_word_eval and "words" in normalized_intervals:
                    for start, end, label in normalized_intervals["words"]:
                        label = label.strip().lower()
                        if not label or label in ["sp", "sil", "<SIL>", "<unk>"]:
                            continue

                        wemb = crop_frames(emb, frame_times, start, end)
                        if len(wemb) >= 4:
                            word2embs[label].append(wemb)

            processed += len(batch_results)

        return {
            "frame_embeddings": all_frame_embs,
            "frame_labels": all_frame_labels,
            "word_embeddings": word2embs,
        }

    def _cluster_pcmi_torch(self, X: torch.Tensor, y: np.ndarray):
        """PCMI clustering + NMI with the batched torch/GPU kernels."""
        if len(X) > self.config.max_frames:

            label_to_indices=defaultdict(list)

            for i,label in enumerate(y):
                label_to_indices[label].append(i)

            freqs={
                label:len(indices)
                for label,indices in label_to_indices.items()
            }

            weights={
                label:np.sqrt(freq)
                for label,freq in freqs.items()
            }

            total_weight=sum(weights.values())
            selected=[]

            for label,indices in label_to_indices.items():

                n=int(self.config.max_frames*weights[label]/total_weight)
                n=min(n,len(indices))

                chosen=np.random.choice(indices, n, replace=False)

                selected.extend(chosen)

            selected=np.array(selected)

            X=X[torch.as_tensor(selected, device=X.device)]
            y=y[selected]

        X = X / torch.clamp(torch.linalg.vector_norm(X, dim=1, keepdim=True), min=1.0)

        cluster_ids = kmeans_fit_predict(
            X, self.config.n_clusters, seed=self.config.kmeans_seed
        ).cpu().numpy()

        nmi_val = float(normalized_mutual_info(y, cluster_ids))
        return nmi_val, cluster_ids, y

    def _cluster_pcmi_legacy(self, X: np.ndarray, y: np.ndarray):
        """PCMI clustering + NMI with the original numpy/sklearn kernels."""
        if len(X)>self.config.max_frames:

            label_to_indices=defaultdict(list)

            for i,label in enumerate(y):
                label_to_indices[label].append(i)

            freqs={
                label:len(indices)
                for label,indices in label_to_indices.items()
            }

            weights={
                label:np.sqrt(freq)
                for label,freq in freqs.items()
            }

            total_weight=sum(weights.values())
            selected=[]

            for label,indices in label_to_indices.items():

                n=int(self.config.max_frames*weights[label]/total_weight)
                n=min(n,len(indices))

                chosen=np.random.choice(indices, n, replace=False)

                selected.extend(chosen)

            selected=np.array(selected)

            X=X[selected]
            y=y[selected]

        norms=np.linalg.norm(X,axis=1,keepdims=True)
        norms[norms==0]=1

        X=(X/norms).astype(np.float16)

        cluster_ids=MiniBatchKMeans(
            n_clusters=self.config.n_clusters,
            batch_size=1000
        ).fit_predict(X)

        nmi_val = float(normalized_mutual_info_score(y, cluster_ids))
        return nmi_val, cluster_ids, y

    def compute_metrics(self, audios:Iterator[np.ndarray], intervals_iter: Iterator[Dict[str, List[Dict[str, Any]]]]) -> dict:
        """Compute evaluation metrics from audio arrays and interval dictionaries.

        Parameters
        ----------
        audios:
            Iterator yielding audio arrays.
        intervals_iter:
            Iterator yielding interval dictionaries of the form
            {"phones": [{"text": ..., "start_time": ..., "end_time": ...}],
             "words": [{"text": ..., "start_time": ..., "end_time": ...}]}

        Returns
        -------
        dict
            A dictionary containing `pcmi_score` when phone-level labels are present
            and `wacs_score` when word-level embeddings are available.
        """

        # -------------------------
        # NMI
        # -------------------------
        data = self.collect_embeddings(audios=audios, intervals_iter=intervals_iter)
        all_embeddings=data["frame_embeddings"]
        all_labels=data["frame_labels"]

        nmi=None

        if len(all_embeddings):

            y=np.array(all_labels)

            if self.backend == "torch":
                X=torch.cat(all_embeddings, dim=0)
                nmi_val, cluster_ids, y = self._cluster_pcmi_torch(X, y)
            else:
                X=np.concatenate(all_embeddings,axis=0)
                nmi_val, cluster_ids, y = self._cluster_pcmi_legacy(X, y)

            nmi= {"pcmi_score": nmi_val}

            label_counts=Counter(y)
            label_probs=np.array(list(label_counts.values()),dtype=np.float64)
            label_probs/=label_probs.sum()

            label_ent=float(entropy(label_probs))
            label_ent_norm=float(label_ent/np.log(len(label_probs)))

            cluster_purity=[]

            for k in range(self.config.n_clusters):

                idx=(cluster_ids==k)

                if idx.sum()==0:
                    continue

                labels_k=y[idx]
                counts=Counter(labels_k)

                purity=max(counts.values())/len(labels_k)
                cluster_purity.append(purity)

            nmi.update({
                "pcmi_normalized_label_entropy":label_ent_norm,
                "pcmi_num_frames":int(len(y)),
                "pcmi_num_labels":int(len(label_counts)),
            })
        # -------------------------
        # WACS
        # -------------------------
        word2embs=data["word_embeddings"]

        pos_sims=[]
        neg_sims=[]

        vocab=[
            w for w,v in word2embs.items()
            if len(v)>=self.config.min_occ
        ]


        vocab=sorted(
            vocab,
            key=lambda x:len(word2embs[x]),
            reverse=True
        )[:self.config.max_words]

        # collect all pair specs first so that the RNG call order is
        # identical between backends; similarities are then computed either
        # all at once (torch) or sequentially (legacy).
        pos_specs=[]
        neg_specs=[]
        for word in vocab:

            embs=word2embs[word]
            pairs=0

            for i in range(len(embs)):
                for j in range(i+1,len(embs)):

                    pos_specs.append((embs[i],embs[j]))

                    pairs+=1

                    if pairs>=self.config.max_pairs:
                        break

                if pairs>=self.config.max_pairs:
                    break

            neg_words=np.random.choice(
                vocab,
                min(5,len(vocab)),
                replace=False
            )

            for nw in neg_words:

                if nw==word:
                    continue

                x=embs[np.random.randint(len(embs))]
                y_=word2embs[nw][
                    np.random.randint(len(word2embs[nw]))
                ]

                neg_specs.append((x,y_))

        if self.backend == "torch":
            for sim in batched_dtw_cosine_sims(pos_specs, device=self.config.device, chunk_size=self.config.dtw_chunk_size):
                if sim is not None:
                    pos_sims.append(sim)
            for sim in batched_dtw_cosine_sims(neg_specs, device=self.config.device, chunk_size=self.config.dtw_chunk_size):
                if sim is not None:
                    neg_sims.append(sim)
        else:
            for x,ys in pos_specs:
                sim=dtw_sim(x,ys)
                if sim is not None:
                    pos_sims.append(sim)
            for x,ys in neg_specs:
                sim=dtw_sim(x,ys)
                if sim is not None:
                    neg_sims.append(sim)

        wacs=None

        if len(pos_sims) and len(neg_sims):

            pos=float(np.mean(pos_sims))
            neg=float(np.mean(neg_sims))

            # ---- WACS enrichments ----
            pos_std=float(np.std(pos_sims))
            neg_std=float(np.std(neg_sims))

            # Cohen's d
            eps=1e-8
            pooled=np.sqrt(
                (
                    ((len(pos_sims)-1)*(pos_std**2))+
                    ((len(neg_sims)-1)*(neg_std**2))
                )/
                max(1,(len(pos_sims)+len(neg_sims)-2))
            )+eps

            cohen_d=(pos-neg)/pooled

            # overlap estimate
            wacs_margin=pos-neg

            wacs={
                "wacs_score":wacs_margin,
                "wacs_positive_dtw":pos,
                "wacs_negative_dtw":neg,
                "wacs_num_positive_pairs":int(len(pos_sims)),
                "wacs_num_negative_pairs":int(len(neg_sims)),
                "wacs_cohen_d":float(cohen_d),
                "wacs_vocab_size": len(vocab)
            }

        if nmi and wacs:
            nmi.update(wacs)
        elif wacs:
            nmi = wacs
        if not nmi:
            warnings.warn("No tiers phones or words", UserWarning)

        return nmi
