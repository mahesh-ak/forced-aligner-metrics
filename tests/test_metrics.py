import os
import itertools
import pytest
import torch
import parselmouth
from parselmouth.praat import call
from datasets import load_dataset
from forced_aligner_metrics import ForcedAlignEval, ForcedAlignEvalConfig

TEST_DATA_ROOT = os.path.join("tests", "data")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def audio_iter_from_items(items):
    for x in items:
        yield x["audio"]["array"][:360000]


def load_textgrid(tg_path, tier_names=["phones", "words"]):
    tg = parselmouth.read(tg_path)
    out = {}
    n_tiers = call(tg, "Get number of tiers")

    for tier_name in tier_names:
        tier_exists = False
        for i in range(1, n_tiers + 1):
            name = call(tg, "Get tier name", i)
            if name == tier_name:
                tier_index = i
                tier_exists = True
                break

        if tier_exists:
            intervals = []
            n_intervals = call(tg, "Get number of intervals", tier_index)

            for j in range(1, n_intervals + 1):
                start = call(tg, "Get start time of interval", tier_index, j)
                end = call(tg, "Get end time of interval", tier_index, j)
                label = call(tg, "Get label of interval", tier_index, j)

                if label.strip() == "":
                    label = "sil"

                intervals.append({'start': start,'end': end, 'text': label})
            out[tier_name] = intervals

    return out

def interval_iter_from_items(items, tg_root):
    for x in items:
        wav_file = os.path.basename(x["audio"]["path"])
        tg_path = os.path.join(tg_root, os.path.splitext(wav_file)[0] + ".TextGrid")
        if not os.path.exists(tg_path):
            yield {}
            continue

        yield load_textgrid(tg_path)


@pytest.mark.integration
@pytest.mark.slow
def test_pcmi_wacs_values():
    lang = "de_de"

    ds = load_dataset(
        os.path.join(TEST_DATA_ROOT, "fleurs.py"),
        lang,
        split="test",
        streaming=True,
        trust_remote_code=True,
    )

    items = list(itertools.islice(ds, 200))

    align_dir = os.path.join(TEST_DATA_ROOT, "alignments", "mms-300m-ipa", lang)

    cfg = ForcedAlignEvalConfig(device=DEVICE)
    evaluator = ForcedAlignEval(cfg)

    metrics = evaluator.compute_metrics(
        audios=audio_iter_from_items(items), intervals_iter=interval_iter_from_items(items, align_dir)
    )

    assert metrics.get("pcmi_score") is not None, "PCMI metrics missing"
    assert metrics.get("wacs_score") is not None, "WACS metrics missing"

    pcmi_value = metrics["pcmi_score"]
    wacs_value = metrics["wacs_score"]

    # expected values with allowed tolerance
    assert pcmi_value == pytest.approx(0.215, abs=0.03)
    assert wacs_value == pytest.approx(0.184, abs=0.03)


@pytest.mark.integration
@pytest.mark.slow
def test_qwen3_fa_wacs_only():
    lang = "ja_jp"

    ds = load_dataset(
        os.path.join(TEST_DATA_ROOT, "fleurs.py"),
        lang,
        split="test",
        streaming=True,
        trust_remote_code=True,
    )

    items = list(itertools.islice(ds, 200))

    align_dir = os.path.join(TEST_DATA_ROOT, "alignments", "qwen3-FA", lang)
    cfg = ForcedAlignEvalConfig(device=DEVICE)
    evaluator = ForcedAlignEval(cfg)

    metrics = evaluator.compute_metrics(
        audios=audio_iter_from_items(items), intervals_iter=interval_iter_from_items(items, align_dir)
    )

    assert metrics.get("pcmi_score") is None, "Expected no PCMI output when phones tier is absent"
    assert metrics.get("wacs_score") is not None, "WACS metrics missing"

    wacs_value = metrics["wacs_score"]
    assert wacs_value == pytest.approx(0.187, abs=0.03)
