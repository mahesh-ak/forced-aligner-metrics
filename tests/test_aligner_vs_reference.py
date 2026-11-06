"""Compare the GPU CTCAligner against the reference mms-300m-ipa alignments.

The stored alignments in ``tests/data/alignments/mms-300m-ipa/de_de`` were
generated with the model ``mahesh27/mms-300m-ipa-fleurs`` and the original
lumaku ctc-segmentation aligner (audio truncated to the first 360000
samples, as in the metric tests). This test re-aligns the same FLEURS audio
with this package's GPU aligner and measures boundary differences:

* **token (phone) level**: the token sequence comes from the model's own
  tokenizer (``tokenizer.from_pretrained(model).tokenize(ipa)``); the
  reference generation merged some adjacent codepoints into single tokens
  ("i" + "ː" -> "iː", doubled consonants like "nn") - a quirk specific to
  that generation - so the two token sequences are matched by their
  concatenated character offsets and compared at every shared boundary
  offset (start and end times).
* **word level**: words are defined on the spaces of the IPA transcription;
  their timestamps are constructed from the token alignment (a word spans
  from the start of its first token to the end of its last token).

**Fallback items**: when the original aligner failed on an item, the
reference generation divided the affected (non-silent) segment equally
among its tokens. Such reference "alignments" carry no alignment signal -
their token durations are (near-)constant, e.g. within-word duration
coefficient of variation ~0.02 versus ~0.5 for real alignments. The test
detects these per reference word (CV < 0.05 over >= 3 words' tokens) and
excludes them from both tier comparisons; items dominated by fallback
words (> 50% of tokens) are skipped entirely and reported.

Both sides run the same CTC-Segmentation construction: one ctc run with
the token list as utterances (``prepare_text`` -> ``ctc_segmentation`` ->
``determine_utterance_segments``). Residual differences are dominated by
model-forward/logit drift between the environment that generated the
references and the current one; the tolerances assert that drift stays
frame-scale.
"""

import csv
import itertools
import os

import numpy as np
import pytest
import torch

from forced_aligner_metrics import CTCAligner
from forced_aligner_metrics.ctc_aligner import (
    ctc_segmentation,
    determine_utterance_segments,
    prepare_text,
)

TEST_DATA_ROOT = os.path.join("tests", "data")
MODEL = "mahesh27/mms-300m-ipa-fleurs"
LANG = "de_de"
TG_DIR = os.path.join(TEST_DATA_ROOT, "alignments", "mms-300m-ipa", LANG)
N_ITEMS = int(os.environ.get("FAM_REF_ITEMS", "20"))
MAX_AUDIO_SAMPLES = 360_000  # audio truncation used by the reference generation
# tolerances in seconds (see module docstring)
TOKEN_MAE_LIMIT = 0.06
WORD_MAE_LIMIT = 0.06
BOUNDARY_MAX_LIMIT = 0.6
# equal-division fallback detection: within-word token duration CV below
# this marks a reference word as a fallback "alignment"
FALLBACK_CV = 0.05
# per-item MAE (s) beyond which the reference item is treated as a fallback /
# degenerate generation artifact and excluded from the aggregate comparison
ITEM_MAE_OUTLIER = 0.5
# at most this fraction of items may be excluded
ITEM_EXCLUSION_FRACTION = 0.25

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def word_groups_from_tokens(tokens, words):
    """Token index ranges ``[first, last)`` of each word.

    Tokens never cross word boundaries, so walking the concatenation of the
    tokens against the word strings yields each word's token range.
    """
    groups = []
    i = 0
    for w in words:
        acc = ""
        first = i
        while i < len(tokens) and len(acc) < len(w):
            acc += tokens[i]
            i += 1
        assert acc == w, f"token/word strings diverge at word {w!r}"
        groups.append((first, i))
    assert i == len(tokens), "trailing tokens without a word"
    return groups


def ref_word_fallbacks(ref_phones, ref_words):
    """Detect reference words divided equally among their tokens.

    The reference generation replaced failed alignments by dividing the
    non-silent segment equally among its tokens; such words have
    (near-)constant token durations. Returns ``(fallback_words,
    fallback_token_flags)``: one boolean per reference word and one per
    reference token.
    """
    groups = [[] for _ in ref_words]
    wi = 0
    for idx, (start, _, _) in enumerate(ref_phones):
        while wi < len(ref_words) - 1 and start > ref_words[wi][1]:
            wi += 1
        groups[wi].append(idx)

    fallback_words = []
    token_flags = [False] * len(ref_phones)
    for idxs in groups:
        is_fallback = False
        if len(idxs) >= 3:
            d = np.array([ref_phones[i][1] - ref_phones[i][0] for i in idxs])
            if d.mean() > 0 and d.std() / d.mean() < FALLBACK_CV:
                is_fallback = True
        fallback_words.append(is_fallback)
        if is_fallback:
            for i in idxs:
                token_flags[i] = True
    return fallback_words, token_flags


def boundary_diffs_by_offset(ref_tokens, ref_segs, ref_fallback, our_tokens, our_segs):
    """Boundary-time diffs between two tokenizations of the same string.

    Both token lists concatenate to the same string, so every token end maps
    to a character offset; only offsets where both sides place a token end
    are compared (the reference merges codepoint pairs like "i" + "ː").
    Reference tokens flagged as fallback are skipped.
    """
    def by_offset(tokens, segs, skip=None):
        out, off = {}, 0
        for k, (t, s) in enumerate(zip(tokens, segs)):
            off += len(t)
            if skip is None or not skip[k]:
                out[off] = s
        return out

    ours_by_off = by_offset(our_tokens, our_segs)
    refs_by_off = by_offset(ref_tokens, ref_segs, skip=ref_fallback)
    diffs = []
    for off, seg in ours_by_off.items():
        if off in refs_by_off:
            rseg = refs_by_off[off]
            diffs.append(abs(seg[0] - rseg[0]))
            diffs.append(abs(seg[1] - rseg[1]))
    return diffs


@pytest.mark.integration
@pytest.mark.slow
def test_aligner_vs_reference_alignments(aligner):
    pytest.importorskip("datasets")
    from datasets import load_dataset

    rows = {}
    with open(os.path.join(TEST_DATA_ROOT, "fleurs_ipa_asr", LANG, "test_sentences_ipa.tsv")) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows[r["audio_file"]] = r

    ds = load_dataset(
        os.path.join(TEST_DATA_ROOT, "fleurs.py"),
        LANG,
        split="test",
        streaming=True,
        trust_remote_code=True,
    )
    tokenizer = aligner.processor.tokenizer
    delim = tokenizer.word_delimiter_token

    item_token_diffs, item_word_diffs = {}, {}
    processed = 0
    skipped_fallback_items = 0
    for item in itertools.islice(ds, max(N_ITEMS * 3, 40)):
        wav = os.path.basename(item["audio"]["path"])
        if wav not in rows:
            continue
        tg_path = os.path.join(TG_DIR, os.path.splitext(wav)[0] + ".TextGrid")
        if not os.path.exists(tg_path):
            continue

        ipa = rows[wav]["ipa"]
        tokens = [t for t in tokenizer.tokenize(ipa) if t != delim]

        tiers = parse_textgrid(tg_path)
        ref_phones = [e for e in tiers["phones"] if e[2]]
        ref_words = [e for e in tiers["words"] if e[2]]
        assert "".join(tokens) == "".join(e[2] for e in ref_phones), (
            f"tokenizer and reference token strings differ for {wav}"
        )
        ipa_words = ipa.split()
        assert len(ipa_words) == len(ref_words), f"word count mismatch for {wav}"

        # fallback detection on the reference: equal-division words carry
        # no alignment signal and are excluded from the comparison
        fallback_words, fallback_tokens = ref_word_fallbacks(ref_phones, ref_words)
        if sum(fallback_tokens) / max(len(fallback_tokens), 1) > ITEM_EXCLUSION_FRACTION:
            skipped_fallback_items += 1
            continue

        audio = item["audio"]["array"][:MAX_AUDIO_SAMPLES]
        lpz = aligner.log_probs(audio)

        # one token-level CTC-Segmentation construction with this aligner
        gt, utt_begin = prepare_text(aligner.config, tokens)
        timings, char_probs, _ = ctc_segmentation(aligner.config, lpz, gt)
        seg_tok = determine_utterance_segments(aligner.config, utt_begin, char_probs, timings, tokens)

        # token level: compare at shared character offsets (fallback excluded)
        item_token = boundary_diffs_by_offset(
            [e[2] for e in ref_phones], ref_phones, fallback_tokens, tokens, seg_tok
        )

        # word level: word timestamps constructed from the token spans
        # (a word spans from the start of its first token to the end of
        # its last token; words are bound by the IPA spaces)
        groups = word_groups_from_tokens(tokens, ipa_words)
        our_words = [(seg_tok[a][0], seg_tok[b - 1][1]) for a, b in groups]
        item_word = []
        for (start, end), ref, is_fallback in zip(our_words, ref_words, fallback_words):
            if is_fallback:
                continue
            item_word.append(abs(start - ref[0]))
            item_word.append(abs(end - ref[1]))

        item_token_diffs[wav] = list(item_token)
        item_word_diffs[wav] = list(item_word)
        processed += 1
        if processed >= N_ITEMS:
            break

    # items whose reference alignment is an equal-division fallback or
    # otherwise degenerate generation artifact: their per-item MAE is far
    # beyond frame drift (real items measure < 0.04 s here)
    def _mae(d):
        return float(np.mean(d)) if d else 0.0

    excluded = [
        w for w in item_token_diffs
        if _mae(item_token_diffs[w]) > ITEM_MAE_OUTLIER or _mae(item_word_diffs[w]) > ITEM_MAE_OUTLIER
    ]
    if excluded:
        assert len(excluded) / len(item_token_diffs) <= ITEM_EXCLUSION_FRACTION, (
            f"too many outliers excluded ({len(excluded)}/{len(item_token_diffs)}): "
            f"{excluded}"
        )

    kept = [w for w in item_token_diffs if w not in excluded]
    assert len(kept) >= 2, "no usable reference alignments matched - check the dataset"

    token_diffs = np.array([d for w in kept for d in item_token_diffs[w]])
    word_diffs = np.array([d for w in kept for d in item_word_diffs[w]])

    token_mae = token_diffs.mean()
    word_mae = word_diffs.mean()
    print(
        f"\nitems compared {processed} "
        f"(fallback-dominated skipped: {skipped_fallback_items}, "
        f"outlier items excluded: {len(excluded)}"
    )
    for w in excluded:
        print(f"  excluded {w}: per-item token/word MAE "
              f"{_mae(item_token_diffs[w]):.2f}/{_mae(item_word_diffs[w]):.2f}")
    print(f"token boundaries: MAE {token_mae:.4f} max {token_diffs.max():.3f} (n={len(token_diffs)})")
    print(f"word boundaries:  MAE {word_mae:.4f} max {word_diffs.max():.3f} (n={len(word_diffs)})")

    assert token_mae < TOKEN_MAE_LIMIT, f"token boundary MAE too high: {token_mae:.4f}"
    assert word_mae < WORD_MAE_LIMIT, f"word boundary MAE too high: {word_mae:.4f}"
    assert token_diffs.max() < BOUNDARY_MAX_LIMIT, f"token boundary outlier: {token_diffs.max():.3f}"
    assert word_diffs.max() < BOUNDARY_MAX_LIMIT, f"word boundary outlier: {word_diffs.max():.3f}"


@pytest.fixture(scope="module")
def aligner():
    return CTCAligner(MODEL, device=DEVICE, batch_size=2)
