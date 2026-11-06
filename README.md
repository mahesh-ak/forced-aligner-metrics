# Forced Alignment Evaluation Metrics: PCMI and WACS

This repository contains the implementation of two reference-free forced alignment evaluation metrics as defined in this [paper](https://aclanthology.org/2026.findings-emnlp.279/):

* **PCMI** (Phoneme-Cluster Mutual Information)
* **WACS** (Word Acoustic Consistency Score)

As of **v1.0.0**, all evaluation kernels (DTW, k-means, NMI, cosine distances) run as batched GPU operations on CUDA devices, and the package additionally ships a **GPU CTC aligner** (`forced_aligner_metrics.CTCAligner`).

## Public API

The package is intended to be used by importing the public interfaces from `forced_aligner_metrics`.

### Configuration

Use `ForcedAlignEvalConfig` to define evaluation behavior. Default config:

```python
from forced_aligner_metrics import ForcedAlignEvalConfig

config = ForcedAlignEvalConfig(
    model="facebook/mms-300m",
    device="cuda",
    layer_range=(15, 16),
    batch_size=2,
    samples_phon_eval=50,
    max_frames=10_000,
    n_clusters=50,
    samples_word_eval=200,
    max_words=200,
    min_occ=3,
    max_pairs=10,
    backend="auto",
)
```

### Main evaluator

Use `ForcedAlignEval` with:

- `audios`: an iterator of audio arrays, each item is a numpy `array` sampled at **16kHz**:
```python
array([0.        , 0.        , 0.        , ..., 0.00074703, 0.00074613, 0.00078106])
```
- `intervals_iter`: an iterator of interval dictionaries, where each item is shaped like:

```python
{
    "phones": [
        {"text": "",  "start": 0.0, "end": 0.2},
        {"text": "h", "start": 0.2, "end": 0.5},
        {"text": "aI", "start": 0.5, "end": 1.0},
    ],
    "words": [
        {"text": "",  "start": 0.0, "end": 0.2},
        {"text": "Hi", "start": 0.2, "end": 1.0},
    ],
}
```
Presence of each of the tiers `"phones"` and `"words"` is optional.

```python
from forced_aligner_metrics import ForcedAlignEval, ForcedAlignEvalConfig

config = ForcedAlignEvalConfig()
evaluator = ForcedAlignEval(config)
metrics = evaluator.compute_metrics(audios=audios, intervals_iter=intervals_iter)
```

The returned metrics dictionary contains the available scores for the evaluated sample. When a `phones` tier is present, the output includes the PCMI score under `pcmi_score`; when word-level embeddings are available, it includes the WACS score under `wacs_score`.

## GPU acceleration (v1.0.0)

When `device` is a CUDA device (or `backend="torch"`), the following run as fully batched torch operations instead of their sequential CPU implementations:

* **DTW (WACS)**: all positive/negative word pairs are evaluated at once by a batched anti-diagonal dynamic program with cosine costs (`torch_ops.batched_dtw_cosine_sims`); the alignment itself is *exact* (the old CPU path used approximate `fastdtw`).
* **Embeddings** stay on the GPU as torch tensors instead of expensive CPU round-trips after every batch.
* **k-means (PCMI)**: deterministic GPU k-means with k-means++ init (`torch_ops.kmeans_fit_predict`, seed configurable via `kmeans_seed`), replacing `MiniBatchKMeans`.
* **NMI (PCMI)**: mutual information from a GPU contingency matrix (`torch_ops.normalized_mutual_info`, arithmetic normalization as in scikit-learn).

The backend is selected via `config.backend`:

* `"auto"` (default): GPU kernels when `device` starts with `"cuda"`, the original numpy/sklearn/fastdtw path otherwise,
* `"torch"`: always use the batched torch kernels (works on CPU tensors too),
* `"legacy"`: always use the original CPU implementation.

The kernels are importable directly for stand-alone use:

```python
from forced_aligner_metrics import batched_dtw_cosine_sims, kmeans_fit_predict, normalized_mutual_info
```

`batched_dtw_cosine_sims(pairs, device, chunk_size)` takes a list of `(x, y)` embedding matrices and returns the pairwise `1 - DTW_distance / path_length` similarity per pair.

## GPU CTC aligner

`CTCAligner` is a GPU implementation of [CTC-Segmentation](https://github.com/lumaku/ctc-segmentation)
([arXiv:2007.09127](https://arxiv.org/abs/2007.09127)): an exhaustive alignment search — no beam search, no approximation — that matches the classic implementation bit-for-bit, with a vectorized table fill handling hour-long audio in seconds. It works with any Hugging Face CTC model; the character list, blank, frame duration and word-boundary symbol are derived automatically.

```python
from forced_aligner_metrics import CTCAligner

aligner = CTCAligner("facebook/wav2vec2-large-960h-lv60-self", device="cuda")

# mono 16 kHz audio + one or several utterances
result = aligner.align(audio, ["hello world", "goodbye world"])
print(result.segments)                 # [(start, end, confidence), ...] per utterance, in seconds
print(result.timings, result.states)   # per-character timings, per-frame character states

# many items at once: batched CTC forward, fast DP per item
results = aligner.align_batch(audios, texts, batch_size=8)
```

### Feeding the metrics

`result.intervals()` returns the alignment in exactly the format `ForcedAlignEval` consumes — `{"text", "start", "end"}` dicts for a `phones` (character) tier and a `words` tier, covering the full timeline:

```python
intervals = result.intervals()
metrics = evaluator.compute_metrics(audios=[audio], intervals_iter=[intervals])
```

### Validation under the package's own metrics

On 200 FLEURS `de_de` items aligned with `mahesh27/mms-300m-ipa-fleurs`, the aligner was judged by the metrics this package provides — PCMI and WACS on GPU - against the stored `mms-300m-ipa` reference alignments, which were generated by the paper's hierarchical word→phone pipeline (words first, then per-word phones):

| alignments | PCMI | WACS |
|---|---|---|
| reference TextGrids (paper's pipeline) | 0.219 | 0.167 |
| **CTCAligner (this package)** | **0.342** | **0.198** |

The single-pass CTC aligner scores comparably-or-higher than the reference under both reference-free metrics; `tests/test_aligner_metrics_gpu.py` asserts these bands on GPU.

## Installation for development

Install the package from PyPI with:

```bash
pip install forced-aligner-metrics
```

The source code is available in the [GitHub repository](https://github.com/mahesh-ak/forced-aligner-metrics).

Install the project in editable mode for development:

```bash
python -m pip install -e '.[test]'
```

## Tests

Run the regression tests with:

```bash
python -m pytest
```

The kernel/aligner unit tests run CPU-only; `-m integration` marks the tests that download datasets and models.

`tests/test_aligner_vs_reference.py` compares the aligner to the stored
`mms-300m-ipa` reference alignments at token and word level
(`FAM_REF_ITEMS`, default 20). `tests/test_aligner_metrics_gpu.py` scores the
aligner by PCMI/WACS on GPU (`FAM_METRIC_ITEMS`, default 200).


## Citation

```
@inproceedings{akavarapu-etal-2026-phoneme,
    title = "Phoneme- and Word-Level Metrics Using Self-Supervised Speech Representations for Forced Alignment Evaluation",
    author = "Akavarapu, V.S.D.S.Mahesh  and
      Daniel, Michael  and
      J{\"a}ger, Gerhard",
    editor = "Lee, Hung-Yi  and
      Dev, Sunipa  and
      Li, Jessy  and
      Elliott, Desmond",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: EMNLP 2026",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-emnlp.279/",
    pages = "5865--5892",
    ISBN = "979-8-89176-462-0",
}
```
