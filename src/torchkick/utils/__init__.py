"""
Utility modules for torchkick.

This package provides common utilities used across the torchkick library:
- Video I/O (reading, writing, progress tracking)
- Visualization (track drawing, detection boxes)
- Timing instrumentation for profiling

Example:
    >>> from torchkick.utils import VideoReader, TrackVisualizer, timed
    >>> 
    >>> @timed
    ... def process_video(path: str):
    ...     with VideoReader(path) as reader:
    ...         for frame in reader:
    ...             yield frame
"""

from torchkick.utils.timing import (
    DEBUG_TIMING,
    timed,
    print_timing_stats,
    reset_timing_stats,
    _timing_stats,
)
from torchkick.utils.video import (
    VideoMetadata,
    VideoReader,
    VideoWriter,
    ProgressTracker,
    generate_output_path,
)
from torchkick.utils.visualization import (
    ColorScheme,
    TrackVisualizer,
    draw_detection_boxes,
)

__all__ = [
    # Timing
    "DEBUG_TIMING",
    "timed",
    "print_timing_stats",
    "reset_timing_stats",
    "_timing_stats",
    # Video
    "VideoMetadata",
    "VideoReader",
    "VideoWriter",
    "ProgressTracker",
    "generate_output_path",
    # Visualization
    "ColorScheme",
    "TrackVisualizer",
    "draw_detection_boxes",
]
