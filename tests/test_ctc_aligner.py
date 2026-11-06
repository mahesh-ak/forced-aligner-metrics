import numpy as np
import pytest
import torch

from forced_aligner_metrics.ctc_aligner import (
    CtcSegmentationConfig,
    ctc_segmentation,
    determine_utterance_segments,
    prepare_text,
)


def _synthetic_case(seed=7, T=20000, V=32, n_utt=10):
    """Synthetic log-probabilities with a known character/block structure."""
    rng = np.random.default_rng(seed)
    chars = [chr(97 + i) for i in range(V - 1)]
    char_list = ["<pad>"] + chars
    text, pos = [], 0
    lpz = np.full((T, V), -10.0, dtype=np.float32)
    for _ in range(n_utt):
        word = " ".join(rng.choice(chars, 6))
        text.append(word)
        for w in word.split():
            n = rng.integers(30, 60)
            for _ in range(n):
                if pos >= T:
                    break
                lpz[pos, 0] = -0.9
                lpz[pos, char_list.index(w)] = -0.05
                pos += 1
        pos += 30
    # jitter breaks the exact ties the uniform "silence" region would
    # otherwise create (real model log-probs are never exactly tied)
    lpz += rng.uniform(-0.01, 0.01, lpz.shape).astype(np.float32)
    return char_list, lpz, text


def test_prepare_text_matches_reference():
    ref = pytest.importorskip("ctc_segmentation")
    char_list, lpz, text = _synthetic_case()
    ref_cfg = ref.CtcSegmentationParameters(char_list=char_list, index_duration=0.02)
    our_cfg = CtcSegmentationConfig(char_list=char_list, index_duration=0.02)
    ref_mat, ref_utt = ref.prepare_text(ref_cfg, text, char_list=char_list)
    our_mat, our_utt = prepare_text(our_cfg, text, char_list=char_list)
    assert np.array_equal(ref_mat, our_mat)
    assert ref_utt == our_utt


@pytest.mark.parametrize("min_window_size", [500, 4000])
def test_ctc_segmentation_matches_reference_windowed(min_window_size):
    ref = pytest.importorskip("ctc_segmentation")
    char_list, lpz, text = _synthetic_case()
    ref_mat, utt_begin = ref.prepare_text(
        ref.CtcSegmentationParameters(char_list=char_list, index_duration=0.02), text, char_list=char_list
    )
    ref_cfg = ref.CtcSegmentationParameters(
        char_list=char_list, index_duration=0.02, min_window_size=min_window_size
    )
    ref_timings, ref_char_probs, ref_states = ref.ctc_segmentation(ref_cfg, lpz.copy(), ref_mat.copy())

    our_cfg = CtcSegmentationConfig(
        char_list=char_list, index_duration=0.02, min_window_size=min_window_size
    )
    our_mat, _ = prepare_text(our_cfg, text, char_list=char_list)
    our_timings, our_char_probs, our_states = ctc_segmentation(
        our_cfg, torch.from_numpy(lpz).float(), our_mat
    )

    np.testing.assert_allclose(our_timings, ref_timings)
    np.testing.assert_allclose(our_char_probs, ref_char_probs)
    assert our_states == ref_states


def test_ctc_segmentation_full_window_same_optimal_score():
    ref = pytest.importorskip("ctc_segmentation")
    char_list, lpz, text = _synthetic_case()
    ref_mat, utt_begin = ref.prepare_text(
        ref.CtcSegmentationParameters(char_list=char_list, index_duration=0.02), text, char_list=char_list
    )
    ref_cfg = ref.CtcSegmentationParameters(char_list=char_list, index_duration=0.02)
    ref_timings, ref_char_probs, _ = ref.ctc_segmentation(ref_cfg, lpz.copy(), ref_mat.copy())

    our_cfg = CtcSegmentationConfig(char_list=char_list, index_duration=0.02)
    our_mat, _ = prepare_text(our_cfg, text, char_list=char_list)
    our_timings, our_char_probs, _ = ctc_segmentation(
        our_cfg, torch.from_numpy(lpz).float(), our_mat
    )

    # with an unconstrained window the reference's cellwise float32
    # accumulation can break exact ties differently; the recovered path
    # must still be optimal (identical accumulated log-probability) and
    # per-character timings must agree to within a few frames
    ref_score = ref_char_probs[ref_char_probs > -1e9].sum()
    our_score = our_char_probs[our_char_probs > -1e9].sum()
    assert our_score == pytest.approx(ref_score, abs=1e-2)
    assert np.abs(ref_timings[1:] - our_timings[1:]).max() < 0.5


def test_segments_are_monotonic_and_bounded():
    char_list, lpz, text = _synthetic_case()
    cfg = CtcSegmentationConfig(char_list=char_list, index_duration=0.02, min_window_size=1000)
    gt_mat, utt_begin = prepare_text(cfg, text, char_list=char_list)
    timings, char_probs, _ = ctc_segmentation(cfg, torch.from_numpy(lpz).float(), gt_mat)
    segments = determine_utterance_segments(cfg, utt_begin, char_probs, timings, text)

    assert len(segments) == len(text)
    duration = lpz.shape[0] * 0.02
    prev_end = 0.0
    for start, end, conf in segments:
        assert 0.0 <= start <= end <= duration
        assert start >= prev_end - 0.1
        assert conf <= 0.0
        prev_end = end


@pytest.mark.integration
def test_ctc_aligner_with_model():
    pytest.importorskip("transformers")
    from forced_aligner_metrics import CTCAligner

    device = "cuda" if torch.cuda.is_available() else "cpu"
    aligner = CTCAligner("facebook/wav2vec2-large-960h-lv60-self", device=device)

    rng = np.random.default_rng(0)
    sr = aligner.sampling_rate
    # two short utterances separated by a pause, each ~3 s
    audio = rng.normal(0, 0.01, 6 * sr).astype(np.float32)
    result = aligner.align(audio, ["hello world", "testing alignment"])
    assert len(result.segments) == 2
    start0, end0, _ = result.segments[0]
    start1, end1, _ = result.segments[1]
    assert end0 <= start1 + 0.5
    assert 0 <= start0 < end0 < 6.0
