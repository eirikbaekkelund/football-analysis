"""
torchkick: Computer vision toolkit for football/soccer video analysis.

A comprehensive library for analyzing football match videos, including:
- Player and ball detection/tracking
- Pitch line detection and camera calibration  
- Homography estimation and pitch projection
- Team identification and jersey color analysis
- Annotation pipeline integration (CVAT)
- SoccerNet dataset utilities

Quick Start:
    >>> import torchkick
    >>> 
    >>> # Read video frames
    >>> with torchkick.VideoReader("match.mp4") as video:
    ...     for frame in video:
    ...         # Process frame
    ...         pass
    >>>
    >>> # Track players
    >>> from torchkick.tracking import SimpleIoUTracker, HomographyEstimator
    >>> tracker = SimpleIoUTracker()
    >>> homography = HomographyEstimator()
    >>>
    >>> # Detect pitch lines
    >>> from torchkick.models import PitchLineDetector
    >>> detector = PitchLineDetector(weights_kp, weights_lines)

Submodules:
    utils: Video I/O, timing, visualization utilities
    soccernet: SoccerNet dataset integration
    annotation: CVAT annotation pipeline
    tracking: Player/ball tracking and pitch projection
    models: Neural network model wrappers
    training: Model training scripts (YOLO, FCNN, RT-DETR, lines)
    inference: Match analysis pipeline

For more examples, see the documentation at https://github.com/torchkick
"""

from torchkick._version import __version__

# Core utilities (available at package root for convenience)
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

# Tracking components commonly used at top level
from torchkick.tracking import (
    SimpleIoUTracker,
    HomographyEstimator,
    PitchVisualizer,
    BallTracker,
)

# Model wrappers
from torchkick.models import (
    PlayerDetector,
    Detection,
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
    "SimpleIoUTracker",
    "HomographyEstimator",
    "PitchVisualizer",
    "BallTracker",
    # Models
    "PlayerDetector",
    "Detection",
    # Inference
    "run_analysis",
]
