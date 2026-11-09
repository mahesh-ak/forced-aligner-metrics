"""GPU-based CTC segmentation aligner.

A reimplementation of the CTC-Segmentation algorithm

    Winkelbauer, Kürzinger, "CTC-Segmentation of Large Corpora for German
    End-to-end Speech Recognition", https://arxiv.org/abs/2007.09127
    (reference implementation: https://github.com/lumaku/ctc-segmentation)

that runs the dynamic-programming alignment on a CUDA device with torch.

The reference implementation fills the alignment table cell-by-cell with a
Cython double loop and does **not** use beam search: it is an exhaustive
(max,+) dynamic program over a sliding window of audio frames per
character.  Here the same recurrence is evaluated one text column at a
time: every column is fully vectorized over the window frames, and the
per-column "stay" recurrence ``x_t = max(x_{t-1} + w_t, a_t)`` (a max-plus
scan) is reduced to a ``cumsum``/``cummax`` pair, so the whole table fill
needs only ``len(text)`` python-level steps even for hour-long audio.

Results match the classic algorithm:

* timings, per-frame character states and per-utterance
  (start, end, confidence) segments are produced in the same format;
* the sliding memory window (``min_window_size``), its data-dependent
  offsets and the doubling retry on failed backtracking are preserved,
  so arbitrarily long audio aligns with bounded GPU memory.

There is no beam and no approximation: the full CTC alignment search is
exhaustive over the window, as in the original.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from transformers import AutoModelForCTC, AutoProcessor

_INF_PROB = -10000000000.0

# ---------------------------------------------------------------------------
# Silence trimming (the paper's reference pipeline), on GPU: word spans are
# trimmed to the locally voiced region via a global energy threshold; the
# trims are propagated to the phones contained in each word. All numeric
# work (energy smoothing, percentile threshold, span search) runs as
# vectorized torch kernels - in one batched pass for many items.
# ---------------------------------------------------------------------------

SILENCE_RATIO = 0.05
MIN_WORD_DUR = 0.04
MIN_PHONE_DUR = 0.02

_SMOOTH_WINDOW = 400
# torch.quantile refuses inputs with more than 2**24 elements; longer rows
# are (deterministically) subsampled for the percentile estimate
_MAX_QUANTILE_ELEMS = 2**24
# budget of the span-gather matrix (spans x chunk width) per GPU pass
_SPAN_GATHER_ELEMS = 2**22

def _smoothed_energy_batch(audios, device):
    """Batched smoothed absolute energy: ``convolve(|x|, ones(400)/400, "same")``.

    GPU port of the paper's ``np.convolve`` smoothing: zero-padded moving
    average, all audios of the batch in one convolution call.

    Parameter
    ---------
    audios:
        Sequence of waveform arrays (each sampled at the same rate).
    device:
        Torch device to run on.

    Returns
    -------
    ``(smooth, lengths)``: the ``(batch, max_len)`` smoothed energies and
    the valid sample count per row.
    """
    lengths = [int(len(a)) for a in audios]
    max_len = max(lengths)
    energy = torch.zeros((len(audios), max_len), dtype=torch.float32, device=device)
    for i, a in enumerate(audios):
        energy[i, : lengths[i]] = torch.as_tensor(
            np.asarray(a, dtype=np.float32), device=device
        )
    energy.abs_()
    kernel = torch.full(
        (1, 1, _SMOOTH_WINDOW), 1.0 / _SMOOTH_WINDOW, device=energy.device
    )
    smooth = torch.nn.functional.conv1d(
        energy.unsqueeze(1), kernel, padding=_SMOOTH_WINDOW // 2
    ).squeeze(1)
    return smooth, lengths


def _global_thresholds(smooth, lengths, factor):
    """Voiced threshold per row: ``factor * quantile(smooth_row, 0.95)`` per row."""
    thrs = []
    for i, n in enumerate(lengths):
        v = smooth[i, :n]
        stride = max(1, math.ceil(v.numel() / _MAX_QUANTILE_ELEMS))
        if stride > 1:
            v = v[::stride]
        thrs.append(float(factor * torch.quantile(v, 0.95)))
    return thrs


def compute_global_threshold(
    audio,
    factor=SILENCE_RATIO,
    device=None,
):
    """Smoothed absolute waveform energy and its global voiced threshold (GPU).

    Parameter
    ---------
    audio:
        Waveform array (16 kHz mono).
    factor:
        Fraction of the smoothed energy's 95th percentile that counts as
        voiced.
    device:
        Torch device for the computation (default CPU).

    Returns
    -------
    ``(smooth, thr)``: the moving-average (400 samples) absolute energy
    as a torch tensor on ``device`` and the global threshold
    ``factor * quantile(smooth, 0.95)``.
    """
    smooth, lengths = _smoothed_energy_batch([np.asarray(audio)], device)
    (thr,) = _global_thresholds(smooth, lengths, factor)
    return smooth[0, : lengths[0]], thr


def trim_interval(
    smooth,
    sr,
    thr,
    start,
    end,
    min_word_dur=MIN_WORD_DUR,
):
    """Trim ``[start, end]`` to the voiced span inside it (single interval).

    ``smooth`` is a 1-D torch tensor as returned by
    :func:`compute_global_threshold`.

    Returns ``(start, end)`` unchanged if the interval holds no voiced
    energy above ``thr`` or if the voiced span is shorter than
    ``min_word_dur``.
    """
    s = int(start * sr)
    e = int(end * sr)

    if e <= s:
        return start, end

    above = smooth[s:e] > thr
    if not bool(above.any()):
        return start, end

    idx0 = int(above.long().argmax())
    idx1 = e - s - 1 - int(above.flip(0).long().argmax())

    ns = (s + idx0) / sr
    ne = (s + idx1) / sr

    if ne - ns < min_word_dur:
        return start, end

    return float(ns), float(ne)


def _trim_spans_batched(
    smooth,
    lengths,
    thrs,
    item_ids,
    starts,
    ends,
    sr,
    min_word_dur,
):
    """Batched :func:`trim_interval` for many (item, span) pairs at once.

    Every word span of the whole batch is trimmed in a single vectorized
    GPU search: the per-span energy chunks are gathered into a
    ``(spans, max_span_len)`` matrix (bounded memory via chunking over
    spans) and the first/last voiced sample per span is found by argmax
    reductions. Only two host syncs: the thresholds and the new bounds.

    Returns
    -------
    ``(new_starts, new_ends)`` lists of floats, aligned with the inputs.
    """
    if len(starts) != len(ends) or len(starts) != len(item_ids):
        raise ValueError("item_ids, starts and ends must have the same length")
    K = len(starts)
    if K == 0:
        return [], []
    B, L = smooth.shape
    device = smooth.device

    item_ids_t = torch.as_tensor(item_ids, dtype=torch.long, device=device)
    starts_t = torch.as_tensor(np.asarray(starts, dtype=np.float64), device=device)
    ends_t = torch.as_tensor(np.asarray(ends, dtype=np.float64), device=device)

    # int(start*sr)-style truncation, as in the reference pipeline
    s_idx = (starts_t * sr).long().clamp_(0, L - 1)
    e_idx = (ends_t * sr).long().clamp_(0, L)
    span_len = (e_idx - s_idx).clamp_(min=0)
    if not bool((span_len > 0).any()):
        return list(map(float, starts)), list(map(float, ends))

    # bound the gather matrix: spans x max_span_len <= _SPAN_GATHER_ELEMS
    max_span = int(span_len.max().item())
    chunk = max(1, _SPAN_GATHER_ELEMS // max_span)

    thr_t = torch.as_tensor(thrs, dtype=torch.float32, device=device)
    flat = smooth.reshape(-1)

    new_s = torch.empty(K, dtype=torch.float64, device=device)
    new_e = torch.empty(K, dtype=torch.float64, device=device)
    for k0 in range(0, K, chunk):
        sl = slice(k0, min(k0 + chunk, K))
        sub_len = span_len[sl]
        m = int(sub_len.max().item())
        if m == 0:
            new_s[sl] = starts_t[sl]
            new_e[sl] = ends_t[sl]
            continue
        offs = torch.arange(m, device=device)
        idx = item_ids_t[sl, None] * L + s_idx[sl, None] + offs[None, :]
        idx = idx.clamp_(max=flat.numel() - 1)
        above = (flat[idx] > thr_t[item_ids_t[sl], None]) & (
            offs[None, :] < sub_len[:, None]
        )
        first = above.long().argmax(dim=1)
        last = m - 1 - above.long().flip(1).argmax(dim=1)
        voiced = above.any(dim=1)
        ns = (s_idx[sl] + first).to(torch.float64) / sr
        ne = (s_idx[sl] + last).to(torch.float64) / sr
        keep = voiced & (ne - ns >= min_word_dur)
        new_s[sl] = torch.where(keep, ns, starts_t[sl])
        new_e[sl] = torch.where(keep, ne, ends_t[sl])

    return new_s.tolist(), new_e.tolist()


def _propagate_word_trims_to_phones(alignment, min_phone_dur, word_key, phone_key):
    """Clip the phones contained in each trimmed word to its new bounds.

    Port of the paper's phone propagation: keeps at least ``min_phone_dur``
    of a clipped first/last phone alive and drops non-positive phones.
    """
    phones = alignment.get(phone_key)
    if not phones:
        return

    for w in alignment[word_key]:

        contained = []

        for ph in phones:

            if ph["text"].strip() == "":
                continue

            overlap = (
                ph["end"] > w["start"]
                and ph["start"] < w["end"]
            )

            if overlap:
                contained.append(ph)

        if not contained:
            continue

        first = contained[0]
        last = contained[-1]

        if first["start"] < w["start"]:

            new_start = min(
                w["start"],
                first["end"] - min_phone_dur,
            )

            if new_start < first["end"]:
                first["start"] = float(new_start)

        if last["end"] > w["end"]:

            new_end = max(
                w["end"],
                last["start"] + min_phone_dur,
            )

            if new_end > last["start"]:
                last["end"] = float(new_end)

    alignment[phone_key] = [
        ph
        for ph in phones
        if ph["end"] > ph["start"]
    ]


def _trim_words_batched(
    alignments,
    smooth,
    lengths,
    thrs,
    sr,
    min_word_dur,
    word_key,
    phone_key,
    min_phone_dur,
):
    """Trim the word tier of many alignments against the batched energies."""
    item_ids, starts, ends = [], [], []
    for b, alignment in enumerate(alignments):
        words = alignment[word_key]
        item_ids.extend(b for _ in words)
        starts.extend(w["start"] for w in words)
        ends.extend(w["end"] for w in words)

    new_starts, new_ends = _trim_spans_batched(
        smooth, lengths, thrs, item_ids, starts, ends, sr, min_word_dur
    )

    pos = 0
    for alignment in alignments:
        for w in alignment[word_key]:
            w["start"] = float(new_starts[pos])
            w["end"] = float(new_ends[pos])
            pos += 1
        _propagate_word_trims_to_phones(alignment, min_phone_dur, word_key, phone_key)


def trim_alignment_silence(
    alignment,
    audio,
    sr=16000,
    silence_ratio=SILENCE_RATIO,
    min_word_dur=MIN_WORD_DUR,
    min_phone_dur=MIN_PHONE_DUR,
    word_key="words",
    phone_key="phones",
    device=None,
):
    """Trim silence from an alignment's intervals (in place, and returned).

    GPU (torch) port of the paper's reference pipeline: every word span is
    shrunk onto its voiced region via the global energy threshold (see
    :func:`compute_global_threshold`); the trims are propagated to the
    phones contained in each word, keeping at least ``min_phone_dur`` of a
    clipped first/last phone alive, and non-positive phones are dropped.

    Parameter
    ---------
    alignment:
        Dict with the two interval tiers, e.g. as returned by
        :meth:`CtcAlignmentResult.intervals`:
        ``{"words": [...], "phones": [...]}``.
    audio:
        The corresponding waveform array sampled at ``sr``.
    sr:
        Sampling rate of ``audio``.
    silence_ratio:
        Global voiced threshold factor.
    min_word_dur:
        Minimum kept word duration in seconds.
    min_phone_dur:
        Minimum kept phone duration in seconds.
    word_key / phone_key:
        Tier names in ``alignment`` (default ``"words"``/``"phones"``).
    device:
        Torch device for the numerics (default CPU).
    """
    smooth, lengths = _smoothed_energy_batch([np.asarray(audio)], device)
    thrs = _global_thresholds(smooth, lengths, silence_ratio)
    _trim_words_batched(
        [alignment],
        smooth,
        lengths,
        thrs,
        sr,
        min_word_dur,
        word_key,
        phone_key,
        min_phone_dur,
    )
    return alignment


def trim_alignment_silence_batch(
    alignments,
    audios,
    sr=16000,
    silence_ratio=SILENCE_RATIO,
    min_word_dur=MIN_WORD_DUR,
    min_phone_dur=MIN_PHONE_DUR,
    word_key="words",
    phone_key="phones",
    device=None,
    batch_size=8,
):
    """Batched :func:`trim_alignment_silence` for many alignments on GPU.

    The audios are smoothed in one batched convolution, the voiced
    thresholds are computed per item and all word spans of a batch are
    trimmed in a single vectorized search per chunk of spans; results are
    equivalent to trimming each alignment on its own.

    Parameter
    ---------
    alignments:
        List of interval dicts, one per audio (modified in place).
    audios:
        The corresponding waveform arrays sampled at ``sr``.
    batch_size:
        Audio items per GPU pass (energy convolution/threshold/trim).

    Returns
    -------
    The ``alignments`` list, with every alignment trimmed in place.
    """
    if len(alignments) != len(audios):
        raise ValueError("alignments and audios must have the same length")
    if len(alignments) == 0:
        return alignments

    bs = int(batch_size) if batch_size else len(alignments)
    for i0 in range(0, len(alignments), bs):
        al_batch = alignments[i0 : i0 + bs]
        au_batch = [np.asarray(a) for a in audios[i0 : i0 + bs]]
        smooth, lengths = _smoothed_energy_batch(au_batch, device)
        thrs = _global_thresholds(smooth, lengths, silence_ratio)
        _trim_words_batched(
            al_batch,
            smooth,
            lengths,
            thrs,
            sr,
            min_word_dur,
            word_key,
            phone_key,
            min_phone_dur,
        )
    return alignments


class CtcSegmentationConfig:
    """Parameters for CTC segmentation (mirrors CtcSegmentationParameters).

    The character set is taken from the ASR model's tokenizer. If the
    character set contains any punctuation characters, ``#``, the Greek
    char "ε", or the space placeholder, adapt these settings.
    """

    max_prob = -10000000000.0
    skip_prob = -10000000000.0
    min_window_size = 8000
    max_window_size = 100000
    index_duration = 0.025
    score_min_mean_over_L = 30
    space = "·"
    blank = 0
    replace_spaces_with_blanks = False
    blank_transition_cost_zero = False
    preamble_transition_cost_zero = True
    backtrack_from_max_t = False
    self_transition = "ε"
    start_of_ground_truth = "#"
    excluded_characters = ".,»«•❍·"
    tokenized_meta_symbol = "▁"
    char_list = None
    # legacy Parameters (ignored; kept for config compatibility)
    subsampling_factor = None
    frame_duration_ms = None

    def __init__(self, **kwargs):
        self.set(**kwargs)

    def set(self, **kwargs):
        """Update parameters; unknown keys or ``None`` values are ignored."""
        for key, value in kwargs.items():
            if not key.startswith("_") and hasattr(self, key) and value is not None:
                setattr(self, key, value)

    @property
    def index_duration_in_seconds(self):
        if self.subsampling_factor and self.frame_duration_ms:
            return self.frame_duration_ms * self.subsampling_factor / 1000
        return self.index_duration

    @property
    def flags(self):
        flags = int(self.blank_transition_cost_zero)
        flags += 2 * int(self.preamble_transition_cost_zero)
        return flags


def prepare_text(config: CtcSegmentationConfig, text: Sequence[str], char_list=None) -> Tuple[np.ndarray, List[int]]:
    """Prepare the given text for CTC segmentation (GPU pipeline compatible).

    Creates a matrix of character symbols to represent the given text,
    then creates a list of char indices depending on the model's char list.

    Parameter
    ---------
    config:
        An instance of CtcSegmentationConfig.
    text:
        Iterable of utterance transcriptions.
    char_list:
        Optional list of all model characters/symbols; characters not
        included in this list are ignored.

    Returns
    -------
    Label matrix of shape ``(n_chars, max_char_len)`` and utterance begin
    indices (position of the space symbol starting each utterance).
    """
    if isinstance(config.blank, str):
        config.blank = 0
    if char_list is not None:
        config.char_list = char_list
    blank = config.char_list[config.blank]
    ground_truth = config.start_of_ground_truth
    utt_begin_indices = []
    for utt in text:
        # One space in-between
        if not ground_truth.endswith(config.space):
            ground_truth += config.space
        # Start new utterance; remember index
        utt_begin_indices.append(len(ground_truth) - 1)
        # Add chars of utterance
        for char in utt:
            if char.isspace() and config.replace_spaces_with_blanks:
                if not ground_truth.endswith(config.space):
                    ground_truth += config.space
            elif char in config.char_list and char not in config.excluded_characters:
                ground_truth += char
    # Add space to the end
    if not ground_truth.endswith(config.space):
        ground_truth += config.space
    utt_begin_indices.append(len(ground_truth) - 1)
    # Create matrix: time frame x number of letters the character symbol spans
    max_char_len = max([len(c) for c in config.char_list])
    ground_truth_mat = np.ones([len(ground_truth), max_char_len], np.int64) * -1
    for i in range(len(ground_truth)):
        for s in range(max_char_len):
            if i - s < 0:
                continue
            span = ground_truth[i - s : i + 1]
            span = span.replace(config.space, blank)
            if span in config.char_list:
                char_index = config.char_list.index(span)
                ground_truth_mat[i, s] = char_index
    return ground_truth_mat, utt_begin_indices


def prepare_tokenized_text(config: CtcSegmentationConfig, text: Sequence[str]) -> Tuple[np.ndarray, List[int]]:
    """Prepare the given tokenized text for CTC segmentation."""
    ground_truth = [config.start_of_ground_truth]
    utt_begin_indices = []
    for utt in text:
        if not ground_truth[-1] == config.space:
            ground_truth += [config.space]
        utt_begin_indices.append(len(ground_truth) - 1)
        for token in utt.split():
            if token in config.char_list:
                if config.replace_spaces_with_blanks and not token.startswith(
                    config.tokenized_meta_symbol
                ):
                    ground_truth += [config.space]
                ground_truth += [token]
    if not ground_truth[-1] == config.space:
        ground_truth += [config.space]
    utt_begin_indices.append(len(ground_truth) - 1)
    max_char_len = 1
    ground_truth_mat = np.ones([len(ground_truth), max_char_len], np.int64) * -1
    for i in range(1, len(ground_truth)):
        if ground_truth[i] == config.space:
            ground_truth_mat[i, 0] = config.blank
        else:
            char_index = config.char_list.index(ground_truth[i])
            ground_truth_mat[i, 0] = char_index
    return ground_truth_mat, utt_begin_indices


def _maxplus_scan(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Vectorized solution of ``x_t = max(x_{t-1} + w_t, a_t)`` with ``x_0 = a_0``.

    The per-frame "stay" recurrence of the CTC trellis is an associative
    (max,+) scan; unpacking it gives
    ``x_t = C_t + max_{u<=t}(a_u - C_u)`` with ``C_t = cumsum(w)_t``,
    expressible as a single ``cumsum`` plus a single ``cummax``.
    """
    csum = torch.cumsum(w, dim=0)
    m = a - csum
    cmax = torch.cummax(m, dim=0).values
    return csum + cmax


def _fill_table(
    lpz: torch.Tensor,
    ground_truth: np.ndarray,
    config: CtcSegmentationConfig,
    window_size: int,
) -> Tuple[torch.Tensor, np.ndarray]:
    """GPU fill of the alignment probability table.

    Port of ``ctc_segmentation_dyn.pyx:cython_fill_table``: one python step
    per text column, fully vectorized over the window frames on the
    tensor device.

    Parameter
    ---------
    lpz:
        ``(n_frames, vocab)`` float tensor of CTC log probabilities.
    ground_truth:
        ``(n_chars, max_char_len)`` int64 label matrix (rows may contain -1).
    config:
        CtcSegmentationConfig instance.
    window_size:
        Length of the sliding frame window per character.

    Returns
    -------
    ``(table, offsets)``: filled ``(window_size, n_chars)`` table tensor and
    the per-character window offset array (in frames).
    """
    device = lpz.device
    T, _ = lpz.shape
    C, S = ground_truth.shape
    W = min(window_size, T)

    gt = torch.from_numpy(ground_truth.astype(np.int64)).to(device)
    lpz = lpz.float()

    prob_max = float(config.max_prob)
    mean_offset = (T - W) / C
    higher_offset = int(mean_offset) + 1

    table = torch.full((W, C), prob_max, dtype=torch.float32, device=device)
    offsets = np.zeros([C], dtype=np.int64)

    idx = torch.arange(W, device=device)
    blank = config.blank

    offset_sum = 0
    last_arg_max = -1
    for c in range(C):
        if c > 0:
            offset = min(
                max(0, last_arg_max - W // 2),
                min(higher_offset, (T - W) - offset_sum),
            )
            offset_sum += offset
        else:
            offset = 0
        offsets[c] = offset_sum
        last_arg_max = -1

        lpz_col = lpz[offset_sum : offset_sum + W]  # (W, V)
        blank_prob = lpz_col[:, blank].double()

        # ---- switch candidates (consumed text positions) ----
        a = torch.full((W,), prob_max, dtype=torch.float64, device=device)
        max_lpz = torch.full((W,), prob_max, dtype=torch.float64, device=device)
        if c > 0:
            for s in range(S):
                char_idx = ground_truth[c, s]
                if char_idx == -1:
                    continue
                src_col = c - 1 - s
                if src_col < 0:
                    # reference would read a wrapped (garbage) column here;
                    # treating the source as unreachable is well-defined and
                    # only differs for multi-char symbols at text positions
                    # where s >= c - has no effect for character token lists.
                    continue
                src = table[:, src_col].double()  # (W,)
                cur_off = (s + 1) * offset
                r = idx - 1 + cur_off
                rvalid = (r >= 0) & (r < W)
                rclamp = torch.clamp(r, min=0, max=W - 1)
                cand = torch.where(rvalid, src[rclamp] + lpz_col[:, char_idx].double(), a)
                a = torch.maximum(a, cand)
                max_lpz = torch.maximum(max_lpz, lpz_col[:, char_idx].double())

        # ---- stay weights ----
        preamble = (config.flags & 2) and c == 0
        stay_zero = (config.flags & 1) and bool((ground_truth[c, :] == 0).any())

        if preamble:
            # stay_prob = 0 for every frame: the whole column is zero
            col = torch.zeros(W, dtype=torch.float64, device=device)
        else:
            if stay_zero:
                w_vec = torch.zeros(W, dtype=torch.float64, device=device)
            else:
                w_vec = torch.maximum(blank_prob, max_lpz)
            if c == 0:
                # reference presets table[0, 0] = 0 and starts the update at t=1;
                # expressed in the scan as a[0] = 0
                a = torch.where(idx == 0, torch.zeros_like(a), a)
            col = _maxplus_scan(a, w_vec)

        table[:, c] = col.to(torch.float32)

        # argmax of the freshly filled column drives the next window offset;
        # torch.argmax returns the first occurrence, matching the reference
        # bookkeeping (first row always sets, then strict improvements only).
        last_arg_max = int(table[:, c].argmax().item())

    return table, offsets


def ctc_segmentation(
    config: CtcSegmentationConfig, lpz: torch.Tensor, ground_truth: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract character-level utterance alignments (torch implementation).

    Parameter
    ---------
    config:
        An instance of CtcSegmentationConfig.
    lpz:
        CTC log probabilities as a ``(n_frames, vocab)`` torch tensor (on
        any torch device - the table fill runs where the tensor lives).
    ground_truth:
        Ground truth label matrix produced by :func:`prepare_text` or
        :func:`prepare_tokenized_text`.

    Returns
    -------
    ``(timings, char_probs, states)`` exactly as the classic implementation:
    per-character timings in seconds, per-frame character log-probabilities
    and the per-frame character states (``self_transition`` for frames where
    the path stays on a position).
    """
    device = lpz.device
    lpz_t = lpz.detach()
    audio_duration = lpz_t.shape[0] * config.index_duration_in_seconds
    if len(ground_truth) > lpz_t.shape[0] and config.skip_prob <= config.max_prob:
        raise AssertionError("Audio is shorter than text!")

    window_size = config.min_window_size
    # Try multiple window lengths if backtracking fails
    while True:
        table, offsets = _fill_table(lpz_t, ground_truth, config, window_size)

        if config.backtrack_from_max_t:
            t = table.shape[0] - 1
        else:
            t = int(table[:, table.shape[1] - 1].argmax().item())
        c = table.shape[1] - 1

        table_np = table.detach().cpu().numpy()
        lpz_np = lpz_t.detach().cpu().numpy()
        if lpz_np.dtype != np.float32:
            # avoid a second full copy when backtracked frames are float32 already
            lpz_np = lpz_np.astype(np.float32)

        # Backtracking (sequential; the DP reconstruction is data-dependent)
        timings = np.zeros([len(ground_truth)])
        char_probs = np.zeros([lpz_t.shape[0]])
        state_list = [""] * lpz_t.shape[0]
        try:
            # Do until start is reached
            while t != 0 or c != 0:
                # Calculate possible transition probs towards the current cell
                min_s = None
                min_switch_prob_delta = np.inf
                max_lpz_prob = config.max_prob
                for s in range(ground_truth.shape[1]):
                    if ground_truth[c, s] != -1:
                        offset = offsets[c] - (offsets[c - 1 - s] if c - s > 0 else 0)
                        switch_prob = (
                            lpz_np[t + offsets[c], ground_truth[c, s]]
                            if c > 0
                            else config.max_prob
                        )
                        est_switch_prob = table_np[t, c] - table_np[t - 1 + offset, c - 1 - s]
                        if abs(switch_prob - est_switch_prob) < min_switch_prob_delta:
                            min_switch_prob_delta = abs(switch_prob - est_switch_prob)
                            min_s = s
                        max_lpz_prob = max(max_lpz_prob, switch_prob)
                stay_prob = (
                    max(lpz_np[t + offsets[c], config.blank], max_lpz_prob)
                    if t > 0
                    else config.max_prob
                )
                est_stay_prob = table_np[t, c] - table_np[t - 1, c]
                # Check which transition has been taken
                if abs(stay_prob - est_stay_prob) > min_switch_prob_delta:
                    # Apply reverse switch transition
                    if c > 0:
                        # Log timing and character - frame alignment
                        for s in range(0, min_s + 1):
                            timings[c - s] = (
                                offsets[c] + t
                            ) * config.index_duration_in_seconds
                        char_probs[offsets[c] + t] = max_lpz_prob
                        char_index = ground_truth[c, min_s]
                        state_list[offsets[c] + t] = config.char_list[char_index]
                    c -= 1 + min_s
                    t -= 1 - offset
                else:
                    # Apply reverse stay transition
                    char_probs[offsets[c] + t] = stay_prob
                    state_list[offsets[c] + t] = config.self_transition
                    t -= 1
        except IndexError:
            window_size *= 2
            if window_size < config.max_window_size:
                print(
                    "IndexError: Backtracking was not successful, "
                    "increasing the window size to: " + str(window_size)
                )
                continue
            else:
                raise
        break
    return timings, char_probs, state_list


def determine_utterance_segments(
    config: CtcSegmentationConfig,
    utt_begin_indices: List[int],
    char_probs: np.ndarray,
    timings: np.ndarray,
    text: Sequence[str],
) -> List[Tuple[float, float, float]]:
    """Utterance-wise alignments from char-wise alignments.

    Parameter
    ---------
    config:
        An instance of CtcSegmentationConfig.
    utt_begin_indices:
        List of time indices of utterance starts.
    char_probs:
        Character positioned probabilities obtained from backtracking.
    timings:
        Mapping of time indices to seconds.
    text:
        List of utterances.

    Returns
    -------
    Segments: a list of utterance start and end times [s] with the
    confidence score of each utterance.
    """

    def compute_time(index, align_type):
        middle = (timings[index] + timings[index - 1]) / 2
        if align_type == "begin":
            return max(timings[index + 1] - 0.5, middle)
        elif align_type == "end":
            return min(timings[index - 1] + 0.5, middle)

    segments = []
    min_prob = np.float64(-10000000000.0)
    for i in range(len(text)):
        start = compute_time(utt_begin_indices[i], "begin")
        end = compute_time(utt_begin_indices[i + 1], "end")
        start_t = int(round(start / config.index_duration_in_seconds))
        end_t = int(round(end / config.index_duration_in_seconds))
        # Compute confidence score by using the min mean probability
        #   after splitting into segments of L frames
        n = config.score_min_mean_over_L
        if end_t <= start_t:
            min_avg = min_prob
        elif end_t - start_t <= n:
            min_avg = np.float64(char_probs[start_t:end_t].mean())
        else:
            min_avg = np.float64(0.0)
            for t in range(start_t, end_t - n):
                min_avg = min(min_avg, char_probs[t : t + n].mean())
        segments.append((start, end, min_avg))
    return segments


@dataclass
class CtcAlignmentResult:
    """Result of a CTC alignment run.

    Attributes
    ----------
    segments:
        One ``(start_seconds, end_seconds, confidence)`` tuple per input
        utterance.
    timings:
        Character-level timing (in seconds) for every ground-truth label.
    char_probs:
        Per-frame character log-probabilities along the aligned path.
    states:
        Per-frame character states along the aligned path
        (``self_transition`` symbol on frames where the path stays).
    utt_begin_indices:
        Index of the ground-truth label that starts each utterance.
    duration:
        Total audio duration in seconds (filled by :class:`CTCAligner`).
    char_symbols:
        Ground-truth symbol (or ``None`` for unalignable rows) per label,
        used to build :meth:`intervals`.
    separator_symbols:
        Symbols treated as word/utterance separators (the space and the
        start-of-ground-truth symbols).
    audio:
        The aligned waveform (16 kHz mono); stored by :class:`CTCAligner`
        so :meth:`intervals` can trim silence by default.
    sampling_rate:
        Sampling rate of ``audio`` (default 16 kHz).
    device:
        Torch device the alignment ran on (the aligner's device); used by
        :meth:`intervals` to run the silence trimming on the same device.
    """

    segments: List[Tuple[float, float, float]]
    timings: np.ndarray
    char_probs: np.ndarray
    states: List[str]
    utt_begin_indices: List[int]
    duration: float = 0.0
    char_symbols: Optional[List[Optional[str]]] = None
    separator_symbols: Tuple[str, ...] = ()
    audio: Optional[np.ndarray] = None
    sampling_rate: int = 16000
    device: Optional[str] = None

    def intervals(
        self,
        char_tier: str = "phones",
        word_tier: str = "words",
        trim_silence: bool = True,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Alignment as interval dicts in the ``ForcedAlignEval`` input format.

        Character symbols become the entries of one tier and consecutive
        non-separator symbols are merged into word entries of a second
        tier. Separator and unalignable rows are emitted with empty
        ``"text"`` (``ForcedAlignEval`` normalizes those to ``"sil"``), so
        the emitted intervals cover the whole audio timeline:

        .. code-block:: python

            {
                "phones": [{"text": "", "start": 0.0, "end": 0.4}, ...],
                "words":  [{"text": "hi", "start": 0.4, "end": 1.0}, ...],
            }

        Parameter
        ---------
        char_tier:
            Key under which the character-level entries are returned
            (default ``"phones"`` so the dict plugs directly into
            ``ForcedAlignEval.compute_metrics``; the CTC units are
            characters, acting as the phone-like symbols of the DP).
        word_tier:
            Key for the word-level entries (default ``"words"``).
        trim_silence:
            Trim word/phone spans onto their voiced region via the
            paper's energy-threshold pipeline
            (:func:`trim_alignment_silence`); **on by default** when the
            result stores its audio (results from
            ``CTCAligner.align()/align_batch()``). Pass ``False`` for the
            raw timeline-covering intervals.

        Returns
        -------
        Dict with the two tiers; every entry is a
        ``{"text": str, "start": float, "end": float}`` dict in seconds.
        """
        if not self.char_symbols:
            raise ValueError(
                "intervals() needs the character symbols produced by "
                "CTCAligner.align()/align_batch() (char_symbols)"
            )
        t = np.asarray(self.timings, dtype=np.float64)
        n = len(self.char_symbols)
        duration = float(self.duration)

        # span of row i: [midpoint(t_{i-1}, t_i), midpoint(t_i, t_{i+1})];
        # rows before the first symbol and after the last extend to the
        # audio bounds so that the entries cover [0, duration].
        starts = np.empty(n)
        ends = np.empty(n)
        starts[0] = 0.0
        if n > 1:
            starts[1:] = (t[:-1] + t[1:]) / 2.0
        ends[:-1] = starts[1:]
        ends[n - 1] = duration
        # guard against tiny timing disorder around multi-char symbols
        starts = np.maximum.accumulate(np.clip(starts, 0.0, duration))
        ends = np.maximum.accumulate(np.clip(ends, starts, duration))

        char_entries: List[Dict[str, Any]] = []
        word_entries: List[Dict[str, Any]] = []
        current: List[Tuple[str, float, float]] = []

        def flush_word():
            if current:
                word_entries.append(
                    {
                        "text": "".join(c for c, _, _ in current),
                        "start": current[0][1],
                        "end": current[-1][2],
                    }
                )
                current.clear()

        for sym, s_, e_ in zip(self.char_symbols, starts.tolist(), ends.tolist()):
            is_separator = sym is None or sym in self.separator_symbols
            text = "" if is_separator else sym
            char_entries.append({"text": text, "start": s_, "end": e_})
            if is_separator:
                flush_word()
                word_entries.append({"text": "", "start": s_, "end": e_})
            else:
                current.append((sym, s_, e_))
        flush_word()

        tiers = {char_tier: char_entries, word_tier: word_entries}

        if trim_silence and self.audio is not None:
            trim_alignment_silence(
                tiers,
                self.audio,
                sr=self.sampling_rate,
                word_key=word_tier,
                phone_key=char_tier,
                device=self.device,
            )

        return tiers


def _symbols_from_ground_truth(
    config: CtcSegmentationConfig, ground_truth: np.ndarray
) -> List[Optional[str]]:
    """The ground-truth symbol per label row (``None`` if not alignable)."""
    symbols: List[Optional[str]] = []
    for i in range(ground_truth.shape[0]):
        symbol = None
        for s in range(ground_truth.shape[1]):
            if ground_truth[i, s] != -1:
                symbol = config.char_list[ground_truth[i, s]]
                break
        symbols.append(symbol)
    return symbols


class CTCAligner:
    """GPU CTC aligner built around a Hugging Face CTC model.

    Runs the CTC forward pass and the alignment DP on ``device`` and returns
    utterance segments with confidence scores, as
    :class:`CtcAlignmentResult`.

    Parameter
    ---------
    model:
        Hugging Face model id of a CTC-based ASR model, e.g.
        ``"facebook/wav2vec2-large-960h-lv60-self"``.
    device:
        Torch device for the model and DP (e.g. ``"cuda"``).
    config:
        Optional pre-built :class:`CtcSegmentationConfig`.
    batch_size:
        Audio items per model forward pass used by :meth:`align_batch`
        (default 8); the alignment DP always runs per item.
    any constructor ``kwargs`` override the respective config attribute (e.g.
    ``space="|"``, ``min_window_size=8000``). The character list, blank
    index and frame duration are always derived from the model.

    Example
    -------
    >>> aligner = CTCAligner("facebook/wav2vec2-large-960h-lv60-self", device="cuda")
    >>> result = aligner.align(audio_16k, ["hello world", "goodbye world"])
    >>> result.segments[0]
    (0.01, 2.53, -0.31)
    """

    def __init__(
        self,
        model: str,
        device: str = "cuda",
        config: Optional[CtcSegmentationConfig] = None,
        batch_size: int = 8,
        **kwargs,
    ):
        self.device = device
        self.batch_size = batch_size
        self.processor = AutoProcessor.from_pretrained(model)
        self.model = AutoModelForCTC.from_pretrained(model).to(device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.sampling_rate = self.processor.feature_extractor.sampling_rate

        vocab: Dict[str, int] = self.processor.tokenizer.get_vocab()
        char_list = [tok for tok, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
        blank_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if blank_id is None:
            blank_id = 0

        # frame duration derived from the convolutional feature encoder
        stride = 1
        for k, s_ in zip(self.model.config.conv_kernel, self.model.config.conv_stride):
            stride *= s_
        index_duration = stride / self.sampling_rate

        overrides = dict(kwargs)

        default_space = next((c for c in ("|", "$", "·", "▁") if c in char_list), None)
        if default_space is not None:
            # use the model's word-boundary symbol for all whitespace so that
            # utterance-internal word boundaries are visible to the DP
            overrides.setdefault("space", default_space)
            overrides.setdefault("replace_spaces_with_blanks", True)

        self.config = config if config is not None else CtcSegmentationConfig()
        self.config.set(index_duration=index_duration, blank=blank_id, char_list=char_list, **overrides)

    def _feat_extract_output_lengths(self, input_lengths) -> np.ndarray:
        """Exact CTC frame counts for the given sample lengths."""
        lengths = np.asarray(input_lengths)
        for kernel_size, stride in zip(self.model.config.conv_kernel, self.model.config.conv_stride):
            lengths = (lengths - kernel_size) // stride + 1
        return lengths

    @torch.no_grad()
    def log_probs_batch(
        self, audios: Sequence[np.ndarray]
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """Batched CTC log-probabilities.

        Parameter
        ---------
        audios:
            List of mono audio arrays at 16 kHz (padded internally).

        Returns
        -------
        ``(lpz, frame_lengths)``: log-probabilities with shape
        ``(batch, max_frames, vocab)`` (padded region is garbage - slice
        per item up to its frame length) and the exact CTC frame count
        per item.
        """
        inputs = self.processor(
            audios, sampling_rate=self.sampling_rate, return_tensors="pt", padding=True
        )
        input_values = inputs.input_values.to(self.device)
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            sample_lengths = attention_mask.sum(-1).cpu().numpy()
        else:
            # pad-aware mask so batched padded inputs match single inputs
            sample_lengths = np.asarray([len(a) for a in audios])
            attention_mask = (
                torch.arange(input_values.shape[1], device=self.device)[None, :]
                < torch.as_tensor(sample_lengths, device=self.device)[:, None]
            ).long()
        logits = self.model(input_values, attention_mask=attention_mask).logits
        lpz = torch.log_softmax(logits.float(), dim=-1)
        frame_lengths = self._feat_extract_output_lengths(sample_lengths)
        return lpz, frame_lengths

    @torch.no_grad()
    def log_probs(self, audio: np.ndarray) -> torch.Tensor:
        """CTC log-probabilities ``(n_frames, vocab)`` for one audio array."""
        lpz, frame_lengths = self.log_probs_batch([audio])
        return lpz[0, : int(frame_lengths[0])]

    def _alignment_from_lpz(
        self, lpz: torch.Tensor, text: Sequence[str], audio: Optional[np.ndarray] = None
    ) -> CtcAlignmentResult:
        """Run text preparation, the GPU DP and segmentation for one item."""
        ground_truth, utt_begin_indices = prepare_text(self.config, text)
        timings, char_probs, states = ctc_segmentation(self.config, lpz, ground_truth)
        segments = determine_utterance_segments(
            self.config, utt_begin_indices, char_probs, timings, text
        )
        return CtcAlignmentResult(
            segments=segments,
            timings=timings,
            char_probs=char_probs,
            states=states,
            utt_begin_indices=utt_begin_indices,
            duration=lpz.shape[0] * self.config.index_duration_in_seconds,
            char_symbols=_symbols_from_ground_truth(self.config, ground_truth),
            # the blank token also marks separators: prepare_text maps the
            # space symbol onto the blank character of the model vocab
            separator_symbols=(
                self.config.space,
                self.config.start_of_ground_truth,
                self.config.char_list[self.config.blank],
            ),
            audio=audio,
            sampling_rate=self.sampling_rate,
            device=self.device,
        )

    def align(
        self, audio: np.ndarray, text: Union[str, Sequence[str]], **kwargs
    ) -> CtcAlignmentResult:
        """Align one audio array with its transcription.

        Parameter
        ---------
        audio:
            Mono audio array at 16 kHz.
        text:
            A single utterance string or a list of utterance strings; the
            result then contains one segment per utterance.

        Returns
        -------
        :class:`CtcAlignmentResult` with utterance segments, character
        timings, per-frame character states and confidences.
        """
        if isinstance(text, str):
            text = [text]

        self.config.set(**kwargs)

        lpz = self.log_probs(audio)
        return self._alignment_from_lpz(lpz, text, audio=np.asarray(audio))

    def align_batch(
        self,
        audios: Sequence[np.ndarray],
        texts: Sequence[Union[str, Sequence[str]]],
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> List[CtcAlignmentResult]:
        """Align many audio/text pairs with batched CTC forward passes.

        The CTC encoder runs on whole batches (padded, mask-aware), then
        the (fast, vectorized) alignment DP is evaluated per item. The
        DP itself is inherently sequential along the text and slides a
        data-dependent window per character, so it stays per item - the
        batching win comes from the model forward, which dominates the
        runtime for short files.

        Parameter
        ---------
        audios:
            Mono audio arrays at 16 kHz.
        texts:
            One text per audio; each text is a single utterance string or
            a list of utterance strings within that audio.
        batch_size:
            Audio items per model forward pass; defaults to the
            constructor's ``batch_size``.

        Returns
        -------
        One :class:`CtcAlignmentResult` per input pair, in input order.
        """
        if len(audios) != len(texts):
            raise ValueError("audios and texts must have the same length")
        if len(audios) == 0:
            return []

        self.config.set(**kwargs)
        bs = self.batch_size if batch_size is None else int(batch_size)

        # length-bucketed batches keep the padded forward efficient: items
        # sorted by audio length, chunked, results restored to input order
        order = sorted(range(len(audios)), key=lambda i: len(audios[i]))
        results: List[Optional[CtcAlignmentResult]] = [None] * len(audios)
        for i0 in range(0, len(audios), bs):
            idx = order[i0 : i0 + bs]
            chunk = [np.asarray(audios[i]) for i in idx]
            lpz_b, frame_lengths = self.log_probs_batch(chunk)
            for j, i in enumerate(idx):
                text = texts[i]
                if isinstance(text, str):
                    text = [text]
                lpz = lpz_b[j, : int(frame_lengths[j])]
                results[i] = self._alignment_from_lpz(lpz, text, audio=chunk[j])
        return results  # type: ignore[return-value]

    def intervals_batch(
        self,
        results: Sequence[CtcAlignmentResult],
        batch_size: Optional[int] = None,
        trim_silence: bool = True,
        char_tier: str = "phones",
        word_tier: str = "words",
    ) -> List[Dict[str, List[Dict[str, Any]]]]:
        """Materialize (and silence-trim) intervals for many results, batched.

        The trimming runs as batched GPU passes - energies smoothed in one
        convolution per audio batch (default: the aligner's
        ``batch_size``, i.e. the same batching as :meth:`align_batch`),
        per-item thresholds and one vectorized span search over all word
        spans of the batch. Equivalent to calling
        ``r.intervals(trim_silence=...)`` per result, but without the
        per-item launch overhead.

        Parameter
        ---------
        results:
            :class:`CtcAlignmentResult` items, e.g. as returned by
            :meth:`align_batch`.
        batch_size:
            Audio items per GPU pass of the trimming (defaults to the
            constructor's ``batch_size``).
        trim_silence:
            Trim word/phone spans onto their voiced region via the
            paper's energy-threshold pipeline; on by default.
        char_tier / word_tier:
            Tier names of the returned dicts (defaults plug directly into
            ``ForcedAlignEval.compute_metrics``).

        Returns
        -------
        One interval dict per result, in input order.
        """
        if len(results) == 0:
            return []
        if trim_silence:
            missing = [i for i, r in enumerate(results) if r.audio is None]
            if missing:
                raise ValueError(
                    f"trim_silence=True requires the audios stored on the "
                    f"results (missing on indices {missing[:5]})"
                )

        tiers_list = [
            r.intervals(
                char_tier=char_tier, word_tier=word_tier, trim_silence=False
            )
            for r in results
        ]
        if trim_silence:
            trim_alignment_silence_batch(
                tiers_list,
                [r.audio for r in results],
                sr=results[0].sampling_rate,
                word_key=word_tier,
                phone_key=char_tier,
                device=self.device,
                batch_size=self.batch_size if batch_size is None else int(batch_size),
            )
        return tiers_list
