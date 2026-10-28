"""Static, learned-retention, and oracle upper-bound baselines."""

from .learned_retention import fit_global_retention, retention_predictions
from .oracle_fusion import oracle_fusion_weights
from .oracle_retention import oracle_retention_weight
from .static_fusion import static_fusion_predictions

__all__ = [
    "fit_global_retention",
    "oracle_fusion_weights",
    "oracle_retention_weight",
    "retention_predictions",
    "static_fusion_predictions",
]

