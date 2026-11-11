"""Public package interface for the forced alignment metrics project.

    from forced_aligner_metrics import ForcedAlignEval, ForcedAlignEvalConfig
    from forced_aligner_metrics import CTCAligner

Low-level modules (evaluation kernels, the aligner internals) stay in their
submodules and are not re-exported here.
"""

from importlib.metadata import PackageNotFoundError, version as _version

from forced_aligner_metrics.eval_aligns import (
    ForcedAlignEval,
    ForcedAlignEvalConfig,
    compute_aas,
)
from forced_aligner_metrics.ctc_aligner import (
    CTCAligner,
    CtcAlignmentResult,
)

try:
    __version__ = _version("forced-aligner-metrics")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0"

__all__ = [
    "ForcedAlignEval",
    "ForcedAlignEvalConfig",
    "compute_aas",
    "CTCAligner",
    "CtcAlignmentResult",
    "__version__",
]
