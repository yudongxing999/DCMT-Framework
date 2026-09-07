"""
DCMT: Dynamic Cross-Modal Tokenization Framework

Adaptive token boundaries integrating human chunking mechanisms
into multimodal LLMs.
"""

from dcmt.model import DCMTModel, DCMTConfig, DCMTOutput
from dcmt.boundary_detector import AdaptiveBoundaryDetector, BoundaryLoss
from dcmt.hierarchical import HierarchicalRepresentation, HierarchicalLevel
from dcmt.alignment import CrossModalAlignment, MutualInformationEstimator

__all__ = [
    "DCMTModel",
    "DCMTConfig",
    "DCMTOutput",
    "AdaptiveBoundaryDetector",
    "BoundaryLoss",
    "HierarchicalRepresentation",
    "HierarchicalLevel",
    "CrossModalAlignment",
    "MutualInformationEstimator",
]

__version__ = "0.1.0"
