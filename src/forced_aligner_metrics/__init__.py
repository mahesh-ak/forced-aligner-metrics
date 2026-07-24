"""Public package interface for the forced alignment metrics project.

Import the main public API directly from this module:

    from forced_aligner_metrics import ForcedAlignEval, ForcedAlignEvalConfig

Use ForcedAlignEvalConfig to configure the evaluator and ForcedAlignEval to
run PCMI/WACS scoring from audio arrays and interval dictionaries.
"""

from importlib.metadata import PackageNotFoundError, version as _version

from forced_aligner_metrics.eval_aligns import (
    ForcedAlignEval,
    ForcedAlignEvalConfig,
)

try:
    __version__ = _version("forced-aligner-metrics")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0"

__all__ = [
    "ForcedAlignEval",
    "ForcedAlignEvalConfig",
    "__version__",
]
