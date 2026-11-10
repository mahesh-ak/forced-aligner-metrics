"""Exact regression: CTCAligner(trim_silence=True) vs the stored -SIL alignments.

Aligns the FLEURS ``de_de`` test split with ``mahesh27/mms-300m-ipa-fleurs``
using this package's ``CTCAligner`` (silence
trimming ON, consecutive batches of 10 over the raw stream, audio truncated
to 360000 samples - the generation recipe) and compares the resulting
intervals against the stored TextGrids in
``tests/data/alignments/mms-300m-ipa-FAM-SIL/de_de``. Empty segments are
removed from the TextGrids before comparing, so both sides are plain
non-empty interval lists. The words tier of the stored TextGrids carries the
orthographic ``word_segmented`` text, so word texts are compared against
that (word spans come from the IPA alignment). Interval counts and texts
must match exactly; timestamps must match within one CTC frame (0.02 s) -
the DP can break ties differently across environments.

**Data**: streamed online exactly like ``test_metrics.py`` via
``load_dataset(tests/data/fleurs.py, "de_de", split="test", streaming=True)``;
the model loads from the hub. ``FAM_REF_ITEMS=<n>`` limits the number of
compared items.
"""

import os

import pytest
import torch

from forced_aligner_metrics import CTCAligner

TEST_DATA_ROOT = os.path.join("tests", "data")
MODEL = "mahesh27/mms-300m-ipa-fleurs"
LANG = "de_de"
TG_DIR = os.path.join(TEST_DATA_ROOT, "alignments", "mms-300m-ipa-FAM-SIL", LANG)
_N = os.environ.get("FAM_REF_ITEMS")
N_ITEMS = int(_N) if _N else None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# timestamps must match within half a CTC frame (0.01 s - the finest DP
# boundary granularity, boundaries being midpoints of char timings);
# counts/texts exactly
FRAME_TOL = 0.01
# the stored alignments were generated with audio truncated to 360000 samples
MAX_AUDIO_SAMPLES = 360_000
# generation used consecutive batches of 10 over the raw stream (batch shape
# changes the padded forward and with it DP tie-breaks)
BATCH_SIZE = 10


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


def _strip_empty(entries):
    return [e for e in entries if e[2] != ""]


def _same_time(a, b):
    # the TextGrid writer rounds to 4 decimals; compare at that precision
    return abs(round(a, 4) - round(b, 4)) <= FRAME_TOL + 1e-9


def _compare_item(wav, result, item, tg_path, use_word_text):
    # half the items are aligned with word_texts= (words tier labeled from
    # word_segmented), the other half without (words tier = joined IPA tokens)
    iv = result.intervals()
    tiers = parse_textgrid(tg_path)
    ipa_words = item["ipa"].split()
    orth_words = item["word_segmented"].split()
    # word counts never diverge between the IPA and word_segmented texts
    assert len(ipa_words) == len(orth_words), f"{wav}: ipa/word_segmented word counts diverge"
    for tier in ("phones", "words"):
        ref = _strip_empty(tiers[tier])
        ours = [e for e in iv[tier] if e["text"] != ""]
        assert len(ref) == len(ours), (
            f"{wav} {tier}: {len(ref)} stored vs {len(ours)} aligned intervals"
        )
        for wi, ((rs, re_, rt), o) in enumerate(zip(ref, ours)):
            assert _same_time(rs, o["start"]), f"{wav} {tier} {rt!r}: start {rs} vs {o['start']}"
            assert _same_time(re_, o["end"]), f"{wav} {tier} {rt!r}: end {re_} vs {o['end']}"
            if tier == "phones":
                assert rt == o["text"], f"{wav} phones: {rt!r} vs {o['text']!r}"
            else:
                if use_word_text:
                    # words tier labeled from word_segmented at align time,
                    # same rule as the generation run -> matches stored text
                    assert o["text"] == rt, f"{wav} words: {o['text']!r} vs {rt!r}"
                elif len(ours) != len(orth_words):
                    # generation kept the joined IPA tokens for this item
                    # (it relabels only when the counts match), and so does
                    # the fallback path
                    assert o["text"] == rt, f"{wav} words: {o['text']!r} vs {rt!r}"
                # else: stored words are orthographic (generation relabeled);
                # the fallback path yields joined tokens - spans/times checked


@pytest.mark.integration
@pytest.mark.slow
def test_aligner_reproduces_stored_sil_alignments(aligner):
    pytest.importorskip("datasets")
    from datasets import load_dataset

    ds = load_dataset(
        os.path.join(TEST_DATA_ROOT, "fleurs.py"), LANG, split="test",
        streaming=True, trust_remote_code=True,
    )

    # batch the raw stream in consecutive BATCH_SIZE chunks, exactly like the
    # generation run (batch_iterator), then compare items with a stored TextGrid
    compared = 0
    seen = 0
    batch = []

    def run_batch(items, base_seen):
        nonlocal compared
        # half the items get word_texts= (words from word_segmented), the
        # other half fall back to joined tokens as word labels
        word_texts = [
            x["word_segmented"] if (base_seen + k) % 2 == 0 else None
            for k, x in enumerate(items)
        ]
        results = aligner.align_batch(
            [x["audio"]["array"][:MAX_AUDIO_SAMPLES] for x in items],
            [x["ipa"] for x in items],
            word_texts=word_texts,
            batch_size=len(items),
        )
        for x, r, wt in zip(items, results, word_texts):
            wav = os.path.basename(x["audio"]["path"])
            tg = os.path.join(TG_DIR, os.path.splitext(wav)[0] + ".TextGrid")
            if not os.path.exists(tg):
                continue
            _compare_item(wav, r, x, tg, use_word_text=wt is not None)
            compared += 1

    for it in ds:
        batch.append(it)
        if len(batch) < BATCH_SIZE:
            continue
        run_batch(batch, seen)
        seen += len(batch)
        batch = []
        if N_ITEMS is not None and compared >= N_ITEMS:
            break
    if batch and (N_ITEMS is None or compared < N_ITEMS):
        run_batch(batch, seen)

    assert compared >= 100, f"only {compared} matched FLEURS items with stored -SIL alignments"
    print(f"\ncompared {compared} items (half word_texts=, half joined-token words): "
          f"all counts/texts exact, timestamps within {FRAME_TOL} s")


@pytest.fixture(scope="module")
def aligner():
    transformers = pytest.importorskip("transformers")
    processor = transformers.AutoProcessor.from_pretrained(MODEL)
    model = transformers.AutoModelForCTC.from_pretrained(MODEL)
    model.load_adapter(LANG)  # language adapter: load BEFORE passing the model to CTCAligner
    return CTCAligner(model, processor, device=DEVICE, batch_size=BATCH_SIZE, trim_silence=True)
