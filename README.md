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
Presence of each of tiers `"phones"` and `"wacs"` is optional.

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

`CTCAligner` is a CTC segmentation aligner (torch reimplementation of [CTC-Segmentation](https://github.com/lumaku/ctc-segmentation), Winkelbauer & Kürzinger, [arXiv:2007.09127](https://arxiv.org/abs/2007.09127)) whose dynamic program runs on the GPU. Like the reference, it is an exhaustive (max,+) alignment search — *no beam search and no approximation* — but the table fill is vectorized: one python step per text column using cumulative (max,+) scans, with the reference's sliding memory window and window-doubling retry preserved for arbitrarily long audio. On real (non-degenerate) inputs it matches the classic implementation bit-for-bit.

```python
from forced_aligner_metrics import CTCAligner

aligner = CTCAligner("facebook/wav2vec2-large-960h-lv60-self", device="cuda")

# mono 16 kHz audio array + one or several utterances
result = aligner.align(audio, ["hello world", "goodbye world"])

print(result.segments)      # [(start, end, confidence), ...] per utterance, seconds
print(result.timings)       # per-character timings (s)
print(result.states)        # per-frame character states along the aligned path
```

### Metrics-format interval output

`result.intervals()` returns the alignment in exactly the format that `ForcedAlignEval` consumes - every entry is a `{"text", "start", "end"}` dict, with one tier built from the character units and one from merged words (both tier names configurable):

```python
intervals = result.intervals()
# {
#   "phones": [{"text": "", "start": 0.0, "end": 0.4}, {"text": "h", ...}, ...],
#   "words":  [{"text": "", "start": 0.0, "end": 0.4}, {"text": "hi", ...}, ...],
# }

# evaluated directly by the metrics
metrics = evaluator.compute_metrics(
    audios=[audio], intervals_iter=[intervals]
)
```

Separator/gap rows (word boundaries, start symbol, dropped characters) are emitted with empty `"text"`, which `ForcedAlignEval` normalizes to `"sil"`; the entries jointly cover `[0, duration]`.

### Batching

The alignment DP itself is sequential along the text (with a data-dependent sliding window per character), so it runs per item; the expensive part - the CTC forward pass - is batched by `align_batch`, which pads, runs one model forward per batch (mask-aware) and then evaluates the fast vectorized DP per item:

```python
audios = [a1, a2, a3]           # mono 16 kHz arrays
texts = ["hello world", "goodbye world", "third utterance"]
results = aligner.align_batch(audios, texts, batch_size=8)   # -> List[CtcAlignmentResult]
```

`CTCAligner(..., batch_size=8)` sets the default; `align_batch(..., batch_size=...)` overrides it per call. `align()` is the single-item shorthand. Items are length-bucketed internally, so the padded batches stay compute-efficient. Measured with `facebook/wav2vec2-large-960h-lv60-self` on an RTX 2080 Ti: many short clips (~3 s) run ~1.4× faster per item through the batched forward, while long compute-bound clips (~20 s) are roughly at parity with sequential `align()` calls - the wav2vec2 encoder already saturates the GPU on such inputs. Very large batches are bounded by GPU memory (the forwards are attention-heavy), so keep `batch_size` moderate (4-16 worked on 11 GB).

The character list, blank index, frame duration and word-boundary symbol are derived automatically from the HF model's tokenizer/feature extractor; every CTC-Segmentation parameter can still be overridden:

```python
aligner = CTCAligner(
    "facebook/wav2vec2-large-960h-lv60-self",
    device="cuda",
    min_window_size=8000,
    index_duration=0.02,   # by default derived from the model
)
```

A model-agnostic low-level API mirrors the classic package for precomputed CTC log-probabilities `lpz` (a `(n_frames, vocab)` torch tensor on any device):

```python
from forced_aligner_metrics import (
    CtcSegmentationConfig, prepare_text, ctc_segmentation, determine_utterance_segments,
)

config = CtcSegmentationConfig(char_list=char_list, index_duration=0.02)
ground_truth, utt_begin_indices = prepare_text(config, text, char_list=char_list)
timings, char_probs, states = ctc_segmentation(config, lpz, ground_truth)
segments = determine_utterance_segments(config, utt_begin_indices, char_probs, timings, text)
```

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

`tests/test_aligner_vs_reference.py` compares the GPU CTC aligner against the stored
`mms-300m-ipa` reference alignments (generated with `mahesh27/mms-300m-ipa-fleurs` and the
original ctc-segmentation aligner) at token and word level; the number of items is
controlled via `FAM_REF_ITEMS` (default 20).


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
