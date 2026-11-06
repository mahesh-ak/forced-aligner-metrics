"""Public package interface for the forced alignment metrics project.

Import the main public API directly from this module:

    from forced_aligner_metrics import ForcedAlignEval, ForcedAlignEvalConfig

Use ForcedAlignEvalConfig to configure the evaluator and ForcedAlignEval to
run PCMI/WACS scoring from audio arrays and interval dictionaries. The
evaluation kernels batched on GPU are exposed as well:

    from forced_aligner_metrics import batched_dtw_cosine_sims, kmeans_fit_predict, normalized_mutual_info

A GPU CTC aligner (torch reimplementation of CTC-Segmentation) is provided
by :class:`forced_aligner_metrics.CTCAligner`.
"""

from importlib.metadata import PackageNotFoundError, version as _version

from forced_aligner_metrics.eval_aligns import (
    ForcedAlignEval,
    ForcedAlignEvalConfig,
)
from forced_aligner_metrics.torch_ops import (
    batched_dtw_cosine_sims,
    kmeans_fit_predict,
    normalized_mutual_info,
)
from forced_aligner_metrics.ctc_aligner import (
    CTCAligner,
    CtcAlignmentResult,
    CtcSegmentationConfig,
    ctc_segmentation,
    determine_utterance_segments,
    prepare_text,
    prepare_tokenized_text,
)

try:
    __version__ = _version("forced-aligner-metrics")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0"

__all__ = [
    "ForcedAlignEval",
    "ForcedAlignEvalConfig",
    "batched_dtw_cosine_sims",
    "kmeans_fit_predict",
    "normalized_mutual_info",
    "CTCAligner",
    "CtcAlignmentResult",
    "CtcSegmentationConfig",
    "ctc_segmentation",
    "determine_utterance_segments",
    "prepare_text",
    "prepare_tokenized_text",
    "__version__",
]
