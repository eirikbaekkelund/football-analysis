"""
torchkick: Computer vision toolkit for football/soccer video analysis.

Quick Start:
    >>> import torchkick
    >>> output = torchkick.run_analysis(
    ...     video_path="match.mp4",
    ...     yolo_weights="weights/best.pt",
    ...     pitch_weights="weights/pitch/best.pt",
    ... )

Submodules:
    utils: Video I/O, timing, visualization utilities
    soccernet: SoccerNet dataset integration
    tracking: Player tracking and pitch projection
    models: Neural network model wrappers
    training: Model training scripts
    inference: Match analysis pipeline
"""

from torchkick._version import __version__

# Core utilities
from torchkick.utils import (
    VideoReader,
    VideoWriter,
    VideoMetadata,
    ProgressTracker,
    generate_output_path,
    TrackVisualizer,
    ColorScheme,
    draw_detection_boxes,
    timed,
    print_timing_stats,
    reset_timing_stats,
)

# Tracking components
from torchkick.tracking import (
    HomographyEstimator,
    PitchVisualizer,
)

# Inference pipeline
from torchkick.inference import run_analysis

__all__ = [
    # Version
    "__version__",
    # Video I/O
    "VideoReader",
    "VideoWriter",
    "VideoMetadata",
    "ProgressTracker",
    "generate_output_path",
    # Visualization
    "TrackVisualizer",
    "ColorScheme",
    "draw_detection_boxes",
    # Timing
    "timed",
    "print_timing_stats",
    "reset_timing_stats",
    # Tracking
    "HomographyEstimator",
    "PitchVisualizer",
    # Inference
    "run_analysis",
]
