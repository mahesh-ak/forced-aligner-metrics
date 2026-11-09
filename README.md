# Forced Alignment Evaluation Metrics: PCMI and WACS

This repository contains the implementation of two reference-free forced alignment evaluation metrics as defined in this [paper](https://aclanthology.org/2026.findings-emnlp.279/):

* **PCMI** (Phoneme-Cluster Mutual Information)
* **WACS** (Word Acoustic Consistency Score)

As of **v1.0.0**, all evaluation kernels (DTW, k-means, NMI, cosine distances) run as batched GPU operations on CUDA devices, and the package additionally ships a **GPU CTC aligner** (`forced_aligner_metrics.CTCAligner`). As of **v1.1.0**, `CTCAligner` can silence-trim its outputs on the GPU with the paper's pipeline; as of **v1.1.1** the trimming runs at align time and the trimmed intervals are stored on every result (opt-in, off by default — see [Feeding the metrics](#feeding-the-metrics)).

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
([arXiv:2007.09127](https://arxiv.org/abs/2007.09127)): an exhaustive alignment search — no beam search, no approximation — that matches the classic implementation bit-for-bit, with a vectorized table fill handling hour-long audio in seconds. It works with any Hugging Face CTC model; the character list, blank, frame duration and word-boundary symbol are derived automatically. The word-boundary symbol is detected from the vocab (`|`, `$`, `·`, `▁`), the tokenizer's `word_delimiter_token`, or by probing `tokenizer.encode("a b")` — so vocabs that use an unusual separator (e.g. `#`) still split words correctly instead of merging the whole utterance into one word.

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

**Silence trimming (v1.1.0/v1.1.1, opt-in — off by default).** The paper's pipeline — word/phone spans are shrunk onto their voiced region via a global energy threshold (`SILENCE_RATIO=0.05` of the 95th percentile of the smoothed `|audio|`, `MIN_WORD_DUR=0.04`, `MIN_PHONE_DUR=0.02`), trims propagated to contained phones — runs as vectorized torch kernels on the aligner's GPU device: batched energy smoothing (`conv1d` with the 400-sample window, numerically identical to the paper's `np.convolve`), per-item `quantile` thresholds, and one gather/argmax search over *all* word spans of a batch at once. Enable it at align time and the **trimmed intervals are stored on every result**, so all accessors agree:

```python
aligner = CTCAligner("mahesh27/mms-300m-ipa-fleurs", device="cuda", trim_silence=True)
result = aligner.align(audio, ["hello world"])     # or align(..., trim_silence=True)
result.intervals()        # trimmed (stored as result.trimmed_intervals)
result.segments           # trimmed utterance spans
result.intervals(trim_silence=False)               # raw timeline-covering intervals

results = aligner.align_batch(audios, texts)       # batched GPU trim per batch
intervals_list = aligner.intervals_batch(results)  # stored trimmed dicts, one per result
```

Without trimming, `intervals()` returns the raw intervals; `intervals(trim_silence=True)` trims on the fly (per item). The kernels are also public for stand-alone use: `trim_alignment_silence` / `trim_alignment_silence_batch` (and `compute_global_threshold`, `trim_interval`).

### Validation under the package's own metrics

On 200 FLEURS `de_de` items aligned with `mahesh27/mms-300m-ipa-fleurs`, the single-pass CTC aligner scores comparably-or-higher than the stored reference TextGrids (paper's hierarchical pipeline) under PCMI/WACS on GPU (0.343 / 0.199 with `trim_silence=True` vs 0.224 / 0.167 reference); `tests/test_aligner_metrics_gpu.py` asserts these bands on GPU.

On the DORECO corpus, alignments from `mahesh27/mms-300m-ipa-doreco` were judged by PCMI/WACS computed with mms-300m embeddings on GPU and by the absolute boundary deviation (AAS, in ms) against the gold alignments:

| Model | PCMI (mms) | WACS (mms) | Phone AAS (ms) | Word AAS (ms) |
| --- | --- | --- | --- | --- |
| gold_doreco | 0.334 ± 0.040 | 0.131 ± 0.029 | 0.0 ± 0.0 (self) | 0.0 ± 0.0 (self) |
| mms-300m-ipa-doreco (paper) | 0.202 ± 0.033 | 0.112 ± 0.024 | 89.3 ± 34.6 | 114.3 ± 51.8 |
| mms-300m-ipa-doreco-SIL (paper) | 0.203 ± 0.033 | 0.114 ± 0.026 | 88.1 ± 32.6 | 106.5 ± 44.1 |
| mms-300m-ipa-doreco-FAM (this package) | 0.301 ± 0.040 | 0.125 ± 0.024 | 65.1 ± 34.5 | 87.4 ± 43.0 |
| mms-300m-ipa-doreco-FAM-SIL (this package) | 0.310 ± 0.040 | 0.129 ± 0.025 | 73.9 ± 40.0 | 111.6 ± 53.4 |

`-SIL` marks the silence-trimmed variants (the trimming pipeline above, enabled with `trim_silence=True` — **not the default**). Silence trimming achieves the best PCMI and WACS (FAM-SIL), but not the best AAS (plain FAM has the lowest phone/word boundary deviation).

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
