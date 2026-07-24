# Forced Alignment Evaluation Metrics: PCMI and WACS

This repository contains the implementation of two reference-free forced alignment evaluation metrics:

* **PCMI** (Phoneme-Cluster Mutual Information)
* **WACS** (Word Acoustic Consistency Score)

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

```python
from forced_aligner_metrics import ForcedAlignEval, ForcedAlignEvalConfig

config = ForcedAlignEvalConfig()
evaluator = ForcedAlignEval(config)
metrics = evaluator.compute_metrics(audios=audios, intervals_iter=intervals_iter)
```

The returned metrics dictionary contains the available scores for the evaluated sample. When a `phones` tier is present, the output includes the PCMI score under `pcmi_score`; when word-level embeddings are available, it includes the WACS score under `wacs_score`.


## Installation for development

Install the project in editable mode for development:

```bash
python -m pip install -e '.[test]'
```

## Tests

Run the regression tests with:

```bash
python -m pytest
```
