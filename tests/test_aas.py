"""Self-tests for the shipped ``compute_aas`` (Average Accumulated Shift).

``compute_aas(x, x)`` must be exactly 0.0 on both tiers - checked on the
DORECO reference alignments (real interval dicts, same format as the other
metrics) and on synthetic examples exercising the silence-span handling
(empty labels and gaps between adjacent intervals become ``"sil"`` spans
that are scored as sequence elements).
"""

import os

import pytest

from forced_aligner_metrics import compute_aas

TEST_DATA_ROOT = os.path.join("tests", "data")
DORECO_PARQUET = os.path.join(TEST_DATA_ROOT, "alignments", "doreco", "anal1239", "validation.parquet")


def _row_to_intervals(row):
    """One DORECO parquet row -> standard interval dict (text/start/end)."""
    return {
        "words": [
            {"text": w["text"], "start": float(w["start"]), "end": float(w["end"])}
            for w in row["words"]
        ],
        "phones": [
            {"text": p["phone"], "start": float(p["start"]), "end": float(p["end"])}
            for p in row["phones"]
        ],
    }


def test_compute_aas_self_on_doreco_is_zero():
    pd = pytest.importorskip("pandas")
    df = pd.read_parquet(DORECO_PARQUET)
    assert len(df) > 0
    for _, row in df.head(25).iterrows():
        iv = _row_to_intervals(row)
        aas = compute_aas(iv, iv)
        assert aas["word_aas_ms"] == 0.0, f"row {row.name}: {aas}"
        assert aas["phone_aas_ms"] == 0.0, f"row {row.name}: {aas}"


def test_compute_aas_silence_spans_scored_as_labels():
    # empty-label intervals become "sil" sequence elements; the
    # self-comparison is still exactly 0 and a shifted copy scores > 0
    gold = {
        "words": [
            {"text": "", "start": 0.0, "end": 0.3},
            {"text": "a", "start": 0.3, "end": 0.5},
            {"text": "", "start": 0.5, "end": 0.7},
            {"text": "b", "start": 0.7, "end": 1.0},
        ],
        "phones": [{"text": "x", "start": 0.3, "end": 1.0}],
    }
    assert compute_aas(gold, gold) == {"word_aas_ms": 0.0, "phone_aas_ms": 0.0}

    shifted = {
        "words": [
            {"text": "a", "start": 0.35, "end": 0.55},
            {"text": "b", "start": 0.75, "end": 1.05},
        ],
        "phones": [{"text": "x", "start": 0.35, "end": 1.05}],
    }
    aas = compute_aas(gold, shifted)
    assert aas["word_aas_ms"] > 0.0
    assert aas["phone_aas_ms"] > 0.0


def test_compute_aas_gap_between_intervals_is_assumed_silence():
    # gold has a 0.2 s gap between "a" and "b" -> a "sil" span is assumed
    # there; pred has none -> the unmatched gold silence contributes its
    # full duration and the shifted "b" boundary its deviation:
    # errs = [0 (a), 0.2 (sil), (|0.7-0.5|+0)/2 = 0.1 (b)] -> mean 0.1 s
    gold = {
        "words": [
            {"text": "a", "start": 0.0, "end": 0.5},
            {"text": "b", "start": 0.7, "end": 1.0},
        ]
    }
    pred = {
        "words": [
            {"text": "a", "start": 0.0, "end": 0.5},
            {"text": "b", "start": 0.5, "end": 1.0},
        ]
    }
    aas = compute_aas(gold, pred)
    assert aas["word_aas_ms"] == pytest.approx(100.0)
    assert aas["phone_aas_ms"] is None


def test_compute_aas_missing_tier_is_none():
    aas = compute_aas({"words": [{"text": "a", "start": 0.0, "end": 1.0}]}, {"phones": []})
    assert aas["word_aas_ms"] is None
    assert aas["phone_aas_ms"] is None
