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
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from transformers import AutoModelForCTC, AutoProcessor

_INF_PROB = -10000000000.0


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
    """

    segments: List[Tuple[float, float, float]]
    timings: np.ndarray
    char_probs: np.ndarray
    states: List[str]
    utt_begin_indices: List[int]


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

    def __init__(self, model: str, device: str = "cuda", config: Optional[CtcSegmentationConfig] = None, **kwargs):
        self.device = device
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

    @torch.no_grad()
    def log_probs(self, audio: np.ndarray) -> torch.Tensor:
        """CTC log-probabilities ``(n_frames, vocab)`` for one audio array."""
        inputs = self.processor(
            audio, sampling_rate=self.sampling_rate, return_tensors="pt"
        )
        input_values = inputs.input_values.to(self.device)
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        logits = self.model(input_values, attention_mask=attention_mask).logits
        return torch.log_softmax(logits[0].float(), dim=-1)

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
        )
