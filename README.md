# Forced Alignment Evaluation Metrics: PCMI and WACS

This repository implements two reference-free forced alignment evaluation metrics from this [paper](https://aclanthology.org/2026.findings-emnlp.279/):

* **PCMI** (Phoneme-Cluster Mutual Information)
* **WACS** (Word Acoustic Consistency Score)

It also contains `CTCAligner`, a GPU implementation of CTC-Segmentation whose output plugs directly into the metrics.

## Public API

```bash
pip install forced-aligner-metrics
```

### Usage

```python
from forced_aligner_metrics import ForcedAlignEval, ForcedAlignEvalConfig

evaluator = ForcedAlignEval(ForcedAlignEvalConfig())  # defaults in the config dataclass
metrics = evaluator.compute_metrics(audios=audios, intervals_iter=intervals_iter)
# metrics["pcmi_score"] with a phones tier, metrics["wacs_score"] with a words tier
```

- `audios`: iterator of mono numpy arrays sampled at 16 kHz
- `intervals_iter`: iterator of interval dicts (either tier optional), each entry `{"text": str, "start": float, "end": float}` in seconds:

```python
{
    "phones": [{"text": "", "start": 0.0, "end": 0.2}, {"text": "h", "start": 0.2, "end": 0.5}, ...],
    "words":  [{"text": "", "start": 0.0, "end": 0.2}, {"text": "Hi", "start": 0.2, "end": 1.0}, ...],
}
```

## GPU CTC aligner

`CTCAligner` is a GPU implementation of [CTC-Segmentation](https://github.com/lumaku/ctc-segmentation) ([arXiv:2007.09127](https://arxiv.org/abs/2007.09127)). It takes the *loaded* model and processor, so model-specific loading (language adapters, dtype, ...) stays with the user:

```python
from transformers import AutoModelForCTC, AutoProcessor
from forced_aligner_metrics import CTCAligner, ForcedAlignEval, ForcedAlignEvalConfig

processor = AutoProcessor.from_pretrained("mahesh27/mms-300m-ipa-fleurs")
model = AutoModelForCTC.from_pretrained("mahesh27/mms-300m-ipa-fleurs")
model.load_adapter("te_in")                           # adapters etc.: load BEFORE passing the model in
aligner = CTCAligner(model, processor, device="cuda")

result = aligner.align(audio, "ellaɾuku namaskaːɾam")  # one audio, one IPA transcription string
intervals = result.intervals()                        # {"phones": [...], "words": [...]} -> ForcedAlignEval input
```

Batched alignment:

```python
audios = [audio1_16k, audio2_16k]
texts = ["ellaɾuku namaskaːɾam", "sɐrʋeːʂaːm sʋɐsti"]  # IPA transcriptions

results = aligner.align_batch(audios, texts, batch_size=8)
intervals_list = aligner.intervals_batch(results)            # one intervals dict per result

# optional: pass the orthographic word-segmented text per audio; when its
# words correspond 1:1 to the IPA words, the words tier carries those words
word_segmented = ["ఎల్లరుకు నమస్కారం", "సర్వేషాం స్వస్తి"]

results = aligner.align_batch(audios, texts, word_texts=word_segmented, batch_size=8)
intervals_list = aligner.intervals_batch(results)
```

Silence trimming is optional — off by default; it runs at align time (batched) and the trimmed intervals are stored on every result:

```python
aligner = CTCAligner(model, processor, device="cuda", trim_silence=True)
results = aligner.align_batch(audios, texts, batch_size=8)
intervals_list = aligner.intervals_batch(results)  # trimmed; result.segments trimmed too
```

## Performance Benchmarks

PCMI/WACS computed with mms-300m embeddings on GPU; AAS = absolute boundary deviation vs gold (ms). `-SIL` = silence-trimmed (`trim_silence=True`).

FLEURS (85 languages):

| Model | PCMI (mms) | WACS (mms) |
| --- | --- | --- |
| MFA | 0.247 ± 0.106 (83/85 langs) | 0.143 ± 0.078 (82/85) |
| mms-300m-ipa (v0.1.0) | 0.239 ± 0.028 | 0.195 ± 0.025 |
| mms-300m-ipa-FAM (v1.1.1+) | 0.370 ± 0.033 | 0.220 ± 0.026 |
| mms-300m-ipa-FAM-SIL (v1.1.1+) | **0.382 ± 0.033** | **0.225 ± 0.027** |

DORECO (45 langauges):

| Model | PCMI (mms) | WACS (mms) | Phone AAS (ms) | Word AAS (ms) |
| --- | --- | --- | --- | --- |
| gold_doreco | 0.334 ± 0.040 | 0.131 ± 0.029 | 0.0 ± 0.0 (self) | 0.0 ± 0.0 (self) |
| mms-300m-ipa-doreco (v0.1.0) | 0.202 ± 0.033 | 0.112 ± 0.024 | 89.3 ± 34.6 | 114.3 ± 51.8 |
| mms-300m-ipa-doreco-SIL (v0.1.0) | 0.203 ± 0.033 | 0.114 ± 0.026 | 88.1 ± 32.6 | 106.5 ± 44.1 |
| mms-300m-ipa-doreco-FAM (v1.1.1+) | 0.301 ± 0.040 | 0.125 ± 0.024 | **65.1 ± 34.5** | **87.4 ± 43.0** |
| mms-300m-ipa-doreco-FAM-SIL (v1.1.1+) | **0.310 ± 0.040** | **0.129 ± 0.025** | 73.9 ± 40.0 | 111.6 ± 53.4 |

-SIL gives the best PCMI/WACS, but not the best AAS (plain FAM is lowest). On FLEURS, -FAM-SIL beats MFA in **75/85 languages** in both PCMI and WACS.

## Installation for development

```bash
git clone https://github.com/mahesh-ak/forced-aligner-metrics.git
cd forced-aligner-metrics
python -m pip install -e '.[test]'
```

## Tests

```bash
python -m pytest
```

`tests/test_aligner_vs_reference.py` aligns the FLEURS `de_de` test split with `CTCAligner(trim_silence=True)` and checks the output against the stored `mms-300m-ipa-FAM-SIL` alignments (counts/texts exact, timestamps within one 0.02 s frame); `FAM_REF_ITEMS=<n>` limits the item count. GPU + dataset required (`-m integration`).

## Version Release Notes

- **v2.0.0** (unreleased): `CTCAligner` takes a loaded CTC model + processor instead of a model id.
- **v1.1.1**: silence trimming runs at align time and is stored on every result (`result.trimmed_intervals`, `result.segments`); opt-in, off by default. Word-boundary symbol auto-detection incl. `#`-style tokenizers.
- **v1.1.0**: GPU silence-trimming pipeline for `CTCAligner` interval outputs.
- **v1.0.0**: all evaluation kernels (DTW, k-means, NMI, cosine) as batched GPU operations; added the GPU CTC aligner.
- **v0.1.0**: initial release.

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
