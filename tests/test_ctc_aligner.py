import numpy as np
import pytest
import torch

from forced_aligner_metrics.ctc_aligner import (
    SILENCE_RATIO,
    MIN_PHONE_DUR,
    MIN_WORD_DUR,
    compute_global_threshold,
    _detect_space_symbol,
    trim_alignment_silence,
    trim_alignment_silence_batch,
    trim_interval,
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


def _result_from_synthetic(cfg_extra=None):
    from forced_aligner_metrics.ctc_aligner import (
        CtcAlignmentResult,
        _symbols_from_ground_truth,
    )

    char_list, lpz, text = _synthetic_case()
    cfg_kwargs = dict(char_list=char_list, index_duration=0.02, min_window_size=1000)
    if cfg_extra:
        cfg_kwargs.update(cfg_extra)
    cfg = CtcSegmentationConfig(**cfg_kwargs)
    gt_mat, utt_begin = prepare_text(cfg, text, char_list=char_list)
    timings, char_probs, states = ctc_segmentation(cfg, torch.from_numpy(lpz).float(), gt_mat)
    segments = determine_utterance_segments(cfg, utt_begin, char_probs, timings, text)
    return CtcAlignmentResult(
        segments=segments,
        timings=timings,
        char_probs=char_probs,
        states=states,
        utt_begin_indices=utt_begin,
        duration=lpz.shape[0] * 0.02,
        char_symbols=_symbols_from_ground_truth(cfg, gt_mat),
        separator_symbols=(cfg.space, cfg.start_of_ground_truth, cfg.char_list[cfg.blank]),
    ), text


def test_intervals_match_metrics_input_format():
    from forced_aligner_metrics.eval_aligns import (
        assign_labels_to_frames,
        normalize_intervals,
    )

    result, text = _result_from_synthetic()
    iv = result.intervals()

    # schema of the ForcedAlignEval input dicts
    assert set(iv.keys()) == {"phones", "words"}
    for tier in ("phones", "words"):
        assert len(iv[tier]) > 0
        for entry in iv[tier]:
            assert set(entry.keys()) == {"text", "start", "end"}
            assert isinstance(entry["text"], str)
            assert entry["start"] <= entry["end"]
            assert 0.0 <= entry["start"] and entry["end"] <= result.duration

    # entries cover the whole timeline monotonically
    for tier in ("phones", "words"):
        starts = [e["start"] for e in iv[tier]]
        ends = [e["end"] for e in iv[tier]]
        prev = 0.0
        for s, e in zip(starts, ends):
            assert s <= e
            assert s >= prev - 1e-6
            prev = e
    assert ends[-1] == pytest.approx(result.duration)
    assert starts[0] == pytest.approx(0.0)

    # words are the joined non-separator characters; separators (spaces
    # mapped to the blank symbol, start symbol, dropped rows) are "" entries.
    # This config keeps replace_spaces_with_blanks=False, so whitespace
    # inside utterances is dropped and each utterance merges to one word.
    words = [e["text"] for e in iv["words"]]
    non_empty = [w for w in words if w]
    assert non_empty == ["".join(utt.split()) for utt in text]

    # the dict plugs directly into the metrics helpers
    normalized = normalize_intervals(iv)
    assert set(normalized.keys()) == {"phones", "words"}
    labels = assign_labels_to_frames(
        np.linspace(0.0, result.duration, 200, endpoint=False),
        normalized["phones"],
    )
    assert len(labels) == 200
    assert all(isinstance(lab, str) for lab in labels)


def test_intervals_tier_names_configurable():
    result, _ = _result_from_synthetic()
    iv = result.intervals(char_tier="chars", word_tier="tokens")
    assert set(iv.keys()) == {"chars", "tokens"}


def test_intervals_without_symbols_raises():
    result, _ = _result_from_synthetic()
    result.char_symbols = None
    with pytest.raises(ValueError):
        result.intervals()


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
    single = aligner.align(audio, ["hello world", "testing alignment"])
    assert len(single.segments) == 2
    start0, end0, _ = single.segments[0]
    start1, end1, _ = single.segments[1]
    assert end0 <= start1 + 0.5
    assert 0 <= start0 < end0 < 6.0

    # batched API: forward pass on the whole batch, DP per item
    audios = [rng.normal(0, 0.01, 4 * sr).astype(np.float32) for _ in range(3)]
    texts = ["one two", "three four", "five six"]
    results = aligner.align_batch(audios, texts, batch_size=2)
    assert len(results) == 3
    for r, a, t in zip(results, audios, texts):
        assert len(r.segments) == 1
        iv = r.intervals()
        assert set(iv.keys()) == {"phones", "words"}
        # non-empty word entries are the (case-folded vocab) input words
        got = [e["text"] for e in iv["words"] if e["text"]]
        assert got == t.upper().split()
        assert r.duration == pytest.approx(len(a) / sr, abs=0.02)

# ---------------------------------------------------------------------------
# silence trimming (GPU port of the paper's reference pipeline)
# ---------------------------------------------------------------------------

SR = 16000


def _tensor_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def _voiced_audio(voiced_spans, duration=3.0, amp=1.0, noise=1e-4):
    """Silence with voiced bursts: sine bursts in the given [(s, e)] spans."""
    rng = np.random.default_rng(3)
    t = np.arange(int(duration * SR)) / SR
    audio = rng.normal(0.0, noise, len(t))
    for s, e in voiced_spans:
        mask = (t >= s) & (t < e)
        audio[mask] += amp * np.sin(2 * np.pi * 100.0 * t[mask])
    return audio.astype(np.float32)


# -----------------------
# numpy reference (the paper's pipeline) for GPU parity
# -----------------------


def _np_compute_global_threshold(audio, factor=SILENCE_RATIO):
    energy = np.abs(audio)
    smooth = np.convolve(energy, np.ones(400) / 400, mode="same")
    thr = factor * np.percentile(smooth, 95)
    return smooth, thr


def _np_trim_interval(smooth, sr, thr, start, end, min_word_dur=MIN_WORD_DUR):
    s = int(start * sr)
    e = int(end * sr)
    if e <= s:
        return start, end
    idx = np.where(smooth[s:e] > thr)[0]
    if len(idx) == 0:
        return start, end
    ns = (s + idx[0]) / sr
    ne = (s + idx[-1]) / sr
    if ne - ns < min_word_dur:
        return start, end
    return float(ns), float(ne)


def _np_trim_alignment_silence(alignment, audio, sr=SR, min_phone_dur=MIN_PHONE_DUR):
    smooth, thr = _np_compute_global_threshold(audio)
    words = alignment["words"]
    phones = alignment["phones"]
    for w in words:
        ns, ne = _np_trim_interval(smooth, sr, thr, w["start"], w["end"])
        w["start"] = float(ns)
        w["end"] = float(ne)
    if len(phones) > 0:
        for w in words:
            contained = [
                ph for ph in phones
                if ph["text"].strip() and ph["end"] > w["start"] and ph["start"] < w["end"]
            ]
            if not contained:
                continue
            first, last = contained[0], contained[-1]
            if first["start"] < w["start"]:
                new_start = min(w["start"], first["end"] - min_phone_dur)
                if new_start < first["end"]:
                    first["start"] = float(new_start)
            if last["end"] > w["end"]:
                new_end = max(w["end"], last["start"] + min_phone_dur)
                if new_end > last["start"]:
                    last["end"] = float(new_end)
        alignment["phones"] = [ph for ph in phones if ph["end"] > ph["start"]]
    return alignment


@pytest.mark.parametrize("device", _tensor_devices())
def test_compute_global_threshold_matches_numpy_reference(device):
    audio = _voiced_audio([(1.0, 1.4)])

    smooth, thr = compute_global_threshold(audio, device=device)

    ref_smooth, ref_thr = _np_compute_global_threshold(audio)
    assert smooth.device.type == torch.device(device).type
    assert smooth.numel() == len(audio)
    np.testing.assert_allclose(
        smooth.cpu().numpy(), ref_smooth, rtol=1e-4, atol=1e-6
    )
    assert thr == pytest.approx(ref_thr, rel=1e-5)


@pytest.mark.parametrize("device", _tensor_devices())
def test_trim_interval_trims_voiced_and_keeps_silent(device):
    audio = _voiced_audio([(1.0, 1.4)])
    smooth, thr = compute_global_threshold(audio, device=device)

    # a word spanning silence around the burst is trimmed onto the burst
    ns, ne = trim_interval(smooth, SR, thr, 0.6, 1.8)
    assert ns > 0.9 and ne < 1.5
    assert ne - ns >= MIN_WORD_DUR

    # a purely silent interval is left untouched
    assert trim_interval(smooth, SR, thr, 0.0, 0.5) == (0.0, 0.5)

    # an inverted/degenerate interval is left untouched
    assert trim_interval(smooth, SR, thr, 1.0, 1.0) == (1.0, 1.0)

    # a voiced span shorter than min_word_dur is left untouched
    tiny = np.zeros(int(SR * 2.0))
    tiny[int(1.0 * SR) : int(1.0 * SR) + 10] = 1.0
    tiny_smooth, _ = compute_global_threshold(tiny, device=device)
    out = trim_interval(tiny_smooth, SR, 0.0, 0.9, 1.1, min_word_dur=MIN_WORD_DUR)
    assert out == (0.9, 1.1)


def _alignment_fixture():
    return {
        "words": [{"text": "hi", "start": 0.6, "end": 1.8}],
        "phones": [
            {"text": "", "start": 0.0, "end": 0.6},
            {"text": "h", "start": 0.6, "end": 1.2},
            {"text": "i", "start": 1.2, "end": 1.8},
            {"text": "", "start": 1.8, "end": 3.0},
        ],
    }


@pytest.mark.parametrize("device", _tensor_devices())
def test_trim_alignment_silence_matches_numpy_reference(device):
    audio = _voiced_audio([(1.0, 1.4)])

    alignment = _alignment_fixture()
    ref = _alignment_fixture()
    out = trim_alignment_silence(alignment, audio, sr=SR, device=device)
    ref = _np_trim_alignment_silence(ref, audio)

    assert out is alignment
    for tier in ("words", "phones"):
        assert len(out[tier]) == len(ref[tier])
        for got, want in zip(out[tier], ref[tier]):
            assert got["text"] == want["text"]
            assert got["start"] == pytest.approx(want["start"], abs=1e-9)
            assert got["end"] == pytest.approx(want["end"], abs=1e-9)

    # the word is trimmed to its voiced region and phones follow
    w = out["words"][0]
    assert w["start"] > 0.9 and w["end"] < 1.5
    h, i = out["phones"][1], out["phones"][2]
    assert h["end"] - h["start"] >= MIN_PHONE_DUR - 1e-9
    assert i["end"] - i["start"] >= MIN_PHONE_DUR - 1e-9


def test_trim_alignment_silence_drops_collapsed_phones():
    audio = _voiced_audio([(1.0, 1.4)])
    alignment = {
        "words": [{"text": "hi", "start": 1.1, "end": 1.3}],
        "phones": [
            # phone collapsed to a point: must be filtered out
            {"text": "h", "start": 1.1, "end": 1.1},
        ],
    }
    out = trim_alignment_silence(alignment, audio, sr=SR)
    assert out["phones"] == []


def _mean_alignment_fixture(seed):
    """Alignment + audio with several words/bursts for batch parity."""
    rng = np.random.default_rng(seed)
    duration = 3.0
    t = np.arange(int(duration * SR)) / SR
    audio = rng.normal(0.0, 1e-4, len(t))
    spans = [(0.2, 0.6), (1.4, 1.9), (2.3, 2.5)]
    for s, e in spans:
        mask = (t >= s) & (t < e)
        audio[mask] += np.sin(2 * np.pi * 100.0 * t[mask])
    audio = audio.astype(np.float32)
    words = [
        {"text": "aa", "start": 0.05, "end": 0.75},
        {"text": "bb", "start": 1.0, "end": 2.2},
        {"text": "cc", "start": 2.4, "end": 2.9},
        {"text": "", "start": 2.9, "end": 3.0},
    ]
    phones = [
        {"text": "a", "start": 0.05, "end": 0.4},
        {"text": "a", "start": 0.4, "end": 0.75},
        {"text": "b", "start": 1.0, "end": 1.6},
        {"text": "b", "start": 1.6, "end": 2.2},
        {"text": "c", "start": 2.4, "end": 2.65},
        {"text": "c", "start": 2.65, "end": 2.9},
        {"text": "", "start": 2.9, "end": 3.0},
    ]
    return {"words": words, "phones": phones}, audio


@pytest.mark.parametrize("device", _tensor_devices())
def test_trim_alignment_silence_batch_matches_numpy_reference(device):
    items = [_mean_alignment_fixture(seed) for seed in (0, 1, 2)]
    alignments = [_mean_alignment_fixture(seed)[0] for seed in (0, 1, 2)]
    audios = [a for _, a in items]
    refs = [_np_trim_alignment_silence(_mean_alignment_fixture(seed)[0], a)
            for seed, (_, a) in zip((0, 1, 2), items)]

    out = trim_alignment_silence_batch(
        alignments, audios, sr=SR, device=device, batch_size=2
    )

    assert out is alignments
    for got, want in zip(alignments, refs):
        for tier in ("words", "phones"):
            assert len(got[tier]) == len(want[tier])
            for g, w in zip(got[tier], want[tier]):
                assert g["text"] == w["text"]
                assert g["start"] == pytest.approx(w["start"], abs=1e-9)
                assert g["end"] == pytest.approx(w["end"], abs=1e-9)


@pytest.mark.parametrize("device", _tensor_devices())
def test_trim_alignment_silence_batch_equals_single(device):
    # per-item trimming and one batched pass must agree exactly
    items = [_mean_alignment_fixture(seed) for seed in (0, 1, 2)]
    audios = [a for _, a in items]

    single = [
        trim_alignment_silence(_mean_alignment_fixture(seed)[0], a, sr=SR, device=device)
        for seed, (_, a) in zip((0, 1, 2), items)
    ]
    batched = trim_alignment_silence_batch(
        [_mean_alignment_fixture(seed)[0] for seed in (0, 1, 2)],
        audios,
        sr=SR,
        device=device,
        batch_size=2,
    )
    for got, want in zip(batched, single):
        for tier in ("words", "phones"):
            for g, w in zip(got[tier], want[tier]):
                assert g["start"] == w["start"]
                assert g["end"] == w["end"]


# -----------------------
# CtcAlignmentResult.intervals / CTCAligner.intervals_batch
# -----------------------


def _result_with_audio(voiced_spans, symbols, timings, duration, sr=SR, device=None):
    from forced_aligner_metrics.ctc_aligner import CtcAlignmentResult

    return CtcAlignmentResult(
        segments=[],
        timings=np.asarray(timings, dtype=np.float64),
        char_probs=np.zeros(1),
        states=[""],
        utt_begin_indices=[],
        duration=duration,
        char_symbols=list(symbols),
        separator_symbols=("#", "·", "<pad>"),
        audio=_voiced_audio(voiced_spans, duration=duration),
        sampling_rate=sr,
        device=device,
    )


def _intervals_setup():
    # shared fixture: "#"/"·" separators with a word "hi" spanning the
    # voiced burst (1.0-1.4 s) plus silence around it
    return (
        [(1.0, 1.4)],
        ["#", "·", "h", "i", "·"],
        [0.0, 0.6, 0.6, 1.8, 1.8],
        3.0,
    )


def test_intervals_raw_by_default_trim_on_request():
    result = _result_with_audio(*_intervals_setup())

    # v1.1.1: no trimming unless requested (or stored at align time)
    raw = result.intervals()
    assert result.intervals(trim_silence=False) == raw

    trimmed = result.intervals(trim_silence=True)
    word_raw = [w for w in raw["words"] if w["text"] == "hi"][0]
    word_trim = [w for w in trimmed["words"] if w["text"] == "hi"][0]
    assert word_raw["start"] == pytest.approx(0.6)
    assert word_trim["start"] > word_raw["start"]
    assert word_trim["end"] < word_raw["end"]
    assert 0.95 < word_trim["start"] < 1.05
    assert 1.35 < word_trim["end"] < 1.45

    # phones stay valid intervals after trimming
    for ph in trimmed["phones"]:
        assert ph["end"] > ph["start"]

    # without audio stored, on-the-fly trimming degrades to the raw timeline
    result.audio = None
    assert result.intervals(trim_silence=True) == raw


def test_intervals_trim_silence_keeps_custom_tier_names():
    result = _result_with_audio(*_intervals_setup())
    trimmed = result.intervals(char_tier="chars", word_tier="tokens", trim_silence=True)
    assert set(trimmed.keys()) == {"chars", "tokens"}
    word_trim = [w for w in trimmed["tokens"] if w["text"] == "hi"][0]
    assert 0.95 < word_trim["start"] < 1.05


def _stub_aligner(device=None, batch_size=8, trim_silence=False):
    # intervals_batch / _apply_trim_to_results wiring test without a model
    from forced_aligner_metrics import CTCAligner

    aligner = object.__new__(CTCAligner)
    aligner.device = device
    aligner.batch_size = batch_size
    aligner.trim_silence = trim_silence
    return aligner


def _result_with_segments(device=None):
    result = _result_with_audio(*_intervals_setup(), device=device)
    result.segments = [(0.6, 1.8, -1.0)]
    return result


@pytest.mark.parametrize("device", _tensor_devices())
def test_aligner_intervals_batch_matches_per_item(device):
    # default: no trimming, batched == per-item raw
    results = [_result_with_audio(*_intervals_setup(), device=device) for _ in range(3)]
    aligner = _stub_aligner(device=device, batch_size=2)

    batched = aligner.intervals_batch(results)
    per_item = [r.intervals() for r in results]

    assert len(batched) == len(per_item)
    for got, want in zip(batched, per_item):
        assert got == want
    assert all(r.trimmed_intervals is None for r in results)


@pytest.mark.parametrize("device", _tensor_devices())
def test_aligner_trim_at_align_time_stored_everywhere(device):
    # trim_silence=True stores trimmed intervals on the result; every
    # accessor then returns the trimmed values, segments included
    results = [_result_with_segments(device=device) for _ in range(3)]
    aligner = _stub_aligner(device=device, batch_size=2, trim_silence=True)

    aligner._apply_trim_to_results(results, [r.audio for r in results])

    for r in results:
        assert r.trimmed_intervals is not None
        # intervals() returns the stored trimmed dict by default
        iv = r.intervals()
        word = [w for w in iv["words"] if w["text"] == "hi"][0]
        assert 0.95 < word["start"] < 1.05
        assert 1.35 < word["end"] < 1.45
        # explicit raw still available
        raw = r.intervals(trim_silence=False)
        word_raw = [w for w in raw["words"] if w["text"] == "hi"][0]
        assert word_raw["start"] == pytest.approx(0.6)
        # segments were trimmed too
        s, e, _c = r.segments[0]
        assert 0.95 < s < 1.05 and 1.35 < e < 1.45
        # mutating the returned dict must not corrupt the store
        iv["words"][0]["start"] = -99.0
        assert r.intervals()["words"][0]["start"] != -99.0

    # intervals_batch returns the stored trimmed dicts (no re-trim needed)
    batched = aligner.intervals_batch(results)
    for got, r in zip(batched, results):
        assert got == r.trimmed_intervals


@pytest.mark.parametrize("device", _tensor_devices())
def test_aligner_intervals_batch_trims_and_stores(device):
    # results aligned without trimming get trimmed (and stored) by
    # intervals_batch(trim_silence=True), matching per-item trimming
    results = [_result_with_segments(device=device) for _ in range(3)]
    aligner = _stub_aligner(device=device, batch_size=2, trim_silence=False)

    batched = aligner.intervals_batch(results, trim_silence=True)
    per_item = [r.intervals(trim_silence=True) for r in results]

    assert all(r.trimmed_intervals is not None for r in results)
    for got, want in zip(batched, per_item):
        for tier in ("words", "phones"):
            for g, w in zip(got[tier], want[tier]):
                assert g["start"] == w["start"]
                assert g["end"] == w["end"]


def test_aligner_intervals_batch_requires_audio():
    results = [_result_with_audio(*_intervals_setup()) for _ in range(2)]
    results[1].audio = None
    aligner = _stub_aligner(device=None, batch_size=2)

    with pytest.raises(ValueError, match="audios"):
        aligner.intervals_batch(results, trim_silence=True)

    # trim_silence=False does not need the audios
    tiers = aligner.intervals_batch(results, trim_silence=False)
    assert len(tiers) == 2


# -----------------------
# word-separator detection ("#" as the vocab's word delimiter)
# -----------------------


class _FakeTokenizer:
    def __init__(self, delim):
        self.word_delimiter_token = delim

    def encode(self, s, add_special_tokens=False):
        # "a b" -> [a, <delim>, b]
        return [0, 1, 2] if s == "a b" else []

    def convert_ids_to_tokens(self, ids):
        table = {0: "a", 1: self.word_delimiter_token, 2: "b"}
        return [table[i] for i in ids]


def test_detect_space_symbol_hash_delimiter():
    char_list = ["<pad>", "#", "a", "b", "c"]
    # usual candidates absent, tokenizer delimiter is "#"
    assert _detect_space_symbol(_FakeTokenizer("#"), char_list) == "#"
    # usual candidate wins when present
    assert _detect_space_symbol(_FakeTokenizer("#"), ["<pad>", "|", "#", "a"]) == "|"
    # nothing detectable -> None
    assert _detect_space_symbol(_FakeTokenizer(None), ["<pad>", "a", "b"]) is None


def test_prepare_text_and_intervals_split_words_on_hash_separator():
    # vocab whose word delimiter is "#" (start_of_ground_truth clash is fine:
    # both are separators in intervals())
    char_list = ["<pad>", "#", "d", "n", "e", "v", "a", "l"]
    cfg = CtcSegmentationConfig(
        char_list=char_list, blank=0, space="#",
        replace_spaces_with_blanks=True, index_duration=0.02,
    )
    gt, utt = prepare_text(cfg, ["dn eval"], char_list=char_list)
    from forced_aligner_metrics.ctc_aligner import (
        CtcAlignmentResult,
        _symbols_from_ground_truth,
    )

    symbols = _symbols_from_ground_truth(cfg, gt)
    # the space between the two words became a separator (blank) row, not dropped
    assert "<pad>" in symbols
    n = len(symbols)
    result = CtcAlignmentResult(
        segments=[],
        timings=np.arange(n, dtype=np.float64) * 0.1,
        char_probs=np.zeros(1),
        states=[""],
        utt_begin_indices=utt,
        duration=n * 0.1,
        char_symbols=symbols,
        separator_symbols=(cfg.space, cfg.start_of_ground_truth, cfg.char_list[cfg.blank]),
    )
    iv = result.intervals()
    words = [w["text"] for w in iv["words"] if w["text"]]
    assert words == ["dn", "eval"]
