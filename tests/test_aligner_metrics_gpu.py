"""PCMI/WACS of CTCAligner-generated alignments vs the reference TextGrids.

The alignment quality of the GPU CTC aligner is judged by the very metrics
this package provides: ``ForcedAlignEval.compute_metrics`` runs on the GPU
(backend "torch") over

1. the stored reference TextGrid intervals (mms-300m-ipa, generated with
   the original pipeline), and
2. the CTC aligner's ``intervals()`` from the single-pass DP.

Measured on 200 de_de FLEURS items (mahesh27/mms-300m-ipa-fleurs):
the reference alignments score PCMI 0.224 / WACS 0.167, and the CTC
aligner's alignments score PCMI 0.343 / WACS 0.199 (with
``trim_silence=True``: silence-trimmed intervals stored at align time) - comparable-or-higher
under both reference-free metrics.

The item count is controlled by ``FAM_METRIC_ITEMS`` (default 200; the
tolerances below are calibrated for the default).
"""

import os

import pytest
import torch

from forced_aligner_metrics import (
    CTCAligner,
    ForcedAlignEval,
    ForcedAlignEvalConfig,
)

TEST_DATA_ROOT = os.path.join("tests", "data")
MODEL = "mahesh27/mms-300m-ipa-fleurs"
LANG = "de_de"
TG_DIR = os.path.join(TEST_DATA_ROOT, "alignments", "mms-300m-ipa", LANG)
N_ITEMS = int(os.environ.get("FAM_METRIC_ITEMS", "200"))
FLEURS_SCRIPT = os.environ.get("FAM_FLEURS_SCRIPT") or os.path.join(TEST_DATA_ROOT, "fleurs.py")
FLEURS_LOCAL_ONLY = os.environ.get("FAM_FLEURS_LOCAL_ONLY", "") == "1"
MAX_AUDIO_SAMPLES = 360_000

# tolerances (absolute deltas of the metrics between the aligner output
# and the reference TextGrid alignments; measured delta: 0.119/0.032)
PLAIN_TOL = (0.15, 0.08)
# sanity band around the published reference-test values
REF_RANGE = {"pcmi": (0.185, 0.245), "wacs": (0.135, 0.215)}


def parse_textgrid(path):
    """Minimal oTextFile IntervalTier reader -> {tier: [(start, end, text)]}."""
    lines = open(path).read().splitlines()
    tiers = {}
    i = 0
    while i < len(lines):
        if lines[i].strip() == '"IntervalTier"':
            name = lines[i + 1].strip().strip('"')
            n = int(lines[i + 4].strip())
            j = i + 5
            entries = []
            for _ in range(n):
                entries.append((float(lines[j]), float(lines[j + 1]), lines[j + 2].strip().strip('"')))
                j += 3
            tiers[name] = entries
            i = j
        else:
            i += 1
    return tiers


@pytest.mark.integration
@pytest.mark.slow
def test_pcmi_wacs_on_aligner_intervals():
    pytest.importorskip("datasets")
    if not torch.cuda.is_available():
        pytest.skip("GPU required: this test validates the GPU metric pipeline")
    from datasets import load_dataset

    dl_config = None
    if FLEURS_LOCAL_ONLY:
        from datasets import DownloadConfig

        dl_config = DownloadConfig(local_files_only=True)
    ds = load_dataset(
        FLEURS_SCRIPT, LANG, split="test", streaming=True,
        trust_remote_code=True, download_config=dl_config,
    )

    items = []
    for it in ds:
        wav = os.path.basename(it["audio"]["path"])
        if not os.path.exists(os.path.join(TG_DIR, os.path.splitext(wav)[0] + ".TextGrid")):
            continue
        items.append(it)
        if len(items) >= N_ITEMS:
            break
    assert len(items) >= 100, "not enough matched FLEURS items"

    audios = [x["audio"]["array"][:MAX_AUDIO_SAMPLES] for x in items]
    texts = [x["ipa"] for x in items]

    # reference intervals from the stored TextGrids
    ref_ivs = []
    for x in items:
        wav = os.path.splitext(os.path.basename(x["audio"]["path"]))[0]
        tiers = parse_textgrid(os.path.join(TG_DIR, wav + ".TextGrid"))
        ref_ivs.append(
            {
                tier: [{"text": t, "start": s, "end": e} for s, e, t in tiers[tier]]
                for tier in ("phones", "words")
            }
        )

    plain = CTCAligner(MODEL, device="cuda", batch_size=8, trim_silence=True)
    plain_ivs = [r.intervals() for r in plain.align_batch(audios, texts)]

    # the evaluator must run the GPU/torch kernels for this test
    cfg = ForcedAlignEvalConfig(device="cuda", backend="torch")
    ev = ForcedAlignEval(cfg)
    assert ev.backend == "torch"

    def metrics_of(ivs):
        out = ev.compute_metrics(audios=iter(list(audios)), intervals_iter=iter(list(ivs)))
        for key in ("pcmi_score", "wacs_score"):
            assert key in out, f"{key} missing (metrics did not run)"
        return out

    m_ref = metrics_of(ref_ivs)
    m_plain = metrics_of(plain_ivs)

    print(f"\nreference PCMI {m_ref['pcmi_score']:.4f} WACS {m_ref['wacs_score']:.4f}")
    print(f"aligner   PCMI {m_plain['pcmi_score']:.4f} WACS {m_plain['wacs_score']:.4f}")

    # sanity: the reference alignments themselves score in the known band
    assert REF_RANGE["pcmi"][0] <= m_ref["pcmi_score"] <= REF_RANGE["pcmi"][1]
    assert REF_RANGE["wacs"][0] <= m_ref["wacs_score"] <= REF_RANGE["wacs"][1]

    # the aligner stays in the same metric regime
    assert abs(m_plain["pcmi_score"] - m_ref["pcmi_score"]) < PLAIN_TOL[0]
    assert abs(m_plain["wacs_score"] - m_ref["wacs_score"]) < PLAIN_TOL[1]
