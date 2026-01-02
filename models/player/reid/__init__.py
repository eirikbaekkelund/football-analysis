"""
Re-Identification (ReID) Models for Football Analysis

Provides models for:
- Player re-identification across camera views
- Team assignment based on appearance
- Referee and ball tracking
- Temporal consistency in tracking
"""

from .reid_model import (
    FootballReIDModel,
    ReIDFeatureExtractor,
    ReIDTrainer,
)
from .appearance_embedder import (
    AppearanceEmbedder,
    TeamClassifier,
)

__all__ = [
    "FootballReIDModel",
    "ReIDFeatureExtractor",
    "ReIDTrainer",
    "AppearanceEmbedder",
    "TeamClassifier",
]
