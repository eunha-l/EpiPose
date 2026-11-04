"""EpiPose: antibody specific epitope prediction by weighting docking poses."""
from .config import Config
from .data import FeatureStore
from .model import EpiPose, to_map

__all__ = ["Config", "FeatureStore", "EpiPose", "to_map"]
__version__ = "1.0.0"
