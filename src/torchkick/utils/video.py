"""
Video I/O utilities for reading and writing video files.

This module provides context managers for safe video file handling with
consistent metadata access and progress tracking.

Example:
    >>> from torchkick.utils import VideoReader, VideoWriter, ProgressTracker
    >>> 
    >>> with VideoReader("input.mp4", max_duration=10) as reader:
    ...     progress = ProgressTracker(reader.max_frames)
    ...     with VideoWriter("output.avi", reader.metadata.fps, 
    ...                      (reader.metadata.width, reader.metadata.height)) as writer:
    ...         for frame in reader:
    ...             processed = process(frame)
    ...             writer.write(processed)
    ...             progress.update()
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np
from pydantic import BaseModel


class VideoMetadata(BaseModel):
    """
    Container for video file metadata.

    Attributes:
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Frames per second.
        total_frames: Total number of frames in the video.
    """

    width: int
    height: int
    fps: int
    total_frames: int

    @property
    def duration(self) -> float:
        """
        Video duration in seconds.

        Returns:
            Duration calculated as total_frames / fps, or 0 if fps is 0.
        """
        return self.total_frames / self.fps if self.fps > 0 else 0.0


class VideoReader:
    """
    Context manager for reading video files with frame iteration.

    Provides safe resource handling, metadata access, and optional duration
    limiting for processing only a portion of the video.

    Args:
        path: Path to the video file.
        max_duration: Maximum duration to read in seconds. None reads full video.

    Raises:
        RuntimeError: If video file cannot be opened.

    Example:
        >>> with VideoReader("match.mp4", max_duration=60) as reader:
        ...     print(f"Processing {reader.max_frames} frames at {reader.metadata.fps} FPS")
        ...     for frame_bgr in reader:
        ...         detect_players(frame_bgr)
    """

    def __init__(self, path: str, max_duration: Optional[float] = None) -> None:
        self.path = path
        self.max_duration = max_duration
        self._cap: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None
        self._max_frames: int = 0
        self._frame_count: int = 0

    def __enter__(self) -> "VideoReader":
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.path}")

        self._metadata = VideoMetadata(
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=int(self._cap.get(cv2.CAP_PROP_FPS)),
            total_frames=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

        if self.max_duration:
            self._max_frames = int(self.max_duration * self._metadata.fps)
        else:
            self._max_frames = self._metadata.total_frames

        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> None:
        if self._cap:
            self._cap.release()

    def __iter__(self) -> Iterator[np.ndarray]:
        """
        Iterate over video frames.

        Yields:
            BGR image as numpy array of shape (height, width, 3).
        """
        self._frame_count = 0
        while self._cap is not None and self._cap.isOpened() and self._frame_count < self._max_frames:
            ret, frame = self._cap.read()
            if not ret:
                break
            self._frame_count += 1
            yield frame

    @property
    def metadata(self) -> VideoMetadata:
        """
        Video metadata (width, height, fps, total_frames).

        Raises:
            RuntimeError: If accessed outside context manager.
        """
        if self._metadata is None:
            raise RuntimeError("VideoReader not opened. Use as context manager.")
        return self._metadata

    @property
    def max_frames(self) -> int:
        """Maximum frames to process (respects max_duration limit)."""
        return self._max_frames

    @property
    def frame_count(self) -> int:
        """Current frame count during iteration."""
        return self._frame_count


class VideoWriter:
    """
    Context manager for writing video files with standard codec settings.

    Uses XVID codec with AVI container by default for cross-platform compatibility.

    Args:
        path: Output file path (extension normalized to .avi for XVID codec).
        fps: Output frame rate.
        size: Frame dimensions as (width, height) tuple.
        codec: FourCC codec string. Default "XVID" for broad compatibility.

    Raises:
        RuntimeError: If video writer cannot be created.

    Example:
        >>> with VideoWriter("output.avi", fps=30, size=(1920, 1080)) as writer:
        ...     for frame in frames:
        ...         writer.write(frame)
        ...     print(f"Wrote {writer.frame_count} frames")
    """

    DEFAULT_CODEC: str = "XVID"
    DEFAULT_EXT: str = ".avi"

    def __init__(
        self,
        path: str,
        fps: int,
        size: Tuple[int, int],
        codec: str = DEFAULT_CODEC,
    ) -> None:
        # Ensure .avi extension for XVID codec
        self._path = Path(path)
        if codec == "XVID" and self._path.suffix.lower() != ".avi":
            self._path = self._path.with_suffix(".avi")
        self.path = str(self._path)

        self.fps = fps
        self.size = size
        self.codec = codec
        self._writer: Optional[cv2.VideoWriter] = None
        self._frame_count: int = 0

    def __enter__(self) -> "VideoWriter":
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self._writer = cv2.VideoWriter(self.path, fourcc, self.fps, self.size)
        if not self._writer.isOpened():
            raise RuntimeError(f"Could not create video writer: {self.path}")
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> None:
        if self._writer:
            self._writer.release()

    def write(self, frame: np.ndarray) -> None:
        """
        Write a frame to the video file.

        Args:
            frame: BGR image as numpy array of shape (height, width, 3).
        """
        if self._writer is None:
            raise RuntimeError("VideoWriter not opened. Use as context manager.")
        self._writer.write(frame)
        self._frame_count += 1

    @property
    def frame_count(self) -> int:
        """Number of frames written so far."""
        return self._frame_count


class ProgressTracker:
    """
    Track processing progress with FPS calculation.

    Provides periodic logging checkpoints and summary statistics for
    long-running video processing tasks.

    Args:
        total_frames: Total number of frames to process.
        log_interval: Number of frames between log messages.

    Example:
        >>> progress = ProgressTracker(total_frames=1000, log_interval=100)
        >>> for frame in frames:
        ...     process(frame)
        ...     progress.update()
        ...     if progress.should_log():
        ...         print(progress.status())
        >>> print(progress.summary())
    """

    def __init__(self, total_frames: int, log_interval: int = 30) -> None:
        self.total_frames = total_frames
        self.log_interval = log_interval
        self._start_time: float = time.time()
        self._frame_count: int = 0
        self._last_log_frame: int = 0

    def update(self, n: int = 1) -> None:
        """
        Update frame count.

        Args:
            n: Number of frames to add to count. Default 1.
        """
        self._frame_count += n

    def should_log(self) -> bool:
        """
        Check if it's time to log progress.

        Returns:
            True if log_interval frames have passed since last log.
        """
        if self._frame_count - self._last_log_frame >= self.log_interval:
            self._last_log_frame = self._frame_count
            return True
        return False

    @property
    def fps(self) -> float:
        """Current average processing FPS."""
        elapsed = time.time() - self._start_time
        return self._frame_count / elapsed if elapsed > 0 else 0.0

    @property
    def percent(self) -> float:
        """Completion percentage (0-100)."""
        return 100.0 * self._frame_count / self.total_frames if self.total_frames > 0 else 0.0

    @property
    def frame_count(self) -> int:
        """Current frame count."""
        return self._frame_count

    def status(self) -> str:
        """
        Status string for logging.

        Returns:
            Formatted string like "[50.0%] Frame 500/1000 | 25.3 FPS"
        """
        return f"[{self.percent:5.1f}%] Frame {self._frame_count}/{self.total_frames} | {self.fps:.1f} FPS"

    def summary(self) -> str:
        """
        Final summary string.

        Returns:
            Formatted string like "Processed 1000 frames in 39.5s (25.3 FPS)"
        """
        elapsed = time.time() - self._start_time
        return f"Processed {self._frame_count} frames in {elapsed:.1f}s ({self.fps:.1f} FPS)"


def generate_output_path(
    input_path: str,
    prefix: str = "output",
    suffix: str = "tracked",
    duration: Optional[float] = None,
) -> str:
    """
    Generate output video path from input path.

    Creates a filename by combining prefix, input stem, optional duration,
    and suffix with underscore separators.

    Args:
        input_path: Input video file path.
        prefix: Output filename prefix.
        suffix: Output filename suffix.
        duration: Duration in seconds (added to filename if provided).

    Returns:
        Output path like "output_matchvideo_10s_tracked.avi"

    Example:
        >>> generate_output_path("match.mp4", duration=60)
        'output_match_60s_tracked.avi'
    """
    input_file = Path(input_path)
    stem = input_file.stem

    parts = [prefix, stem]
    if duration:
        parts.append(f"{int(duration)}s")
    parts.append(suffix)

    output_name = "_".join(parts) + ".avi"
    return output_name


__all__ = [
    "VideoMetadata",
    "VideoReader",
    "VideoWriter",
    "ProgressTracker",
    "generate_output_path",
]
