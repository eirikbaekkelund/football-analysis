import cv2
import time
import numpy as np
from pathlib import Path
from pydantic import BaseModel
from typing import Iterator, Optional, Tuple


class VideoMetadata(BaseModel):
    """Video metadata container."""

    width: int
    height: int
    fps: int
    total_frames: int

    @property
    def duration(self) -> float:
        """Video duration in seconds."""
        return self.total_frames / self.fps if self.fps > 0 else 0


class VideoReader:
    """
    Context manager for reading video files with consistent metadata access.

    Usage:
        with VideoReader("video.mp4", max_duration=10) as reader:
            for frame_bgr in reader:
                process(frame_bgr)
    """

    def __init__(self, path: str, max_duration: Optional[float] = None):
        """
        Args:
            path: Path to video file
            max_duration: Maximum duration to read in seconds (None = full video)
        """
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

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._cap:
            self._cap.release()

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over frames (BGR format)."""
        self._frame_count = 0
        while self._cap.isOpened() and self._frame_count < self._max_frames:
            ret, frame = self._cap.read()
            if not ret:
                break
            self._frame_count += 1
            yield frame

    @property
    def metadata(self) -> VideoMetadata:
        """Video metadata (width, height, fps, total_frames)."""
        if self._metadata is None:
            raise RuntimeError("VideoReader not opened. Use as context manager.")
        return self._metadata

    @property
    def max_frames(self) -> int:
        """Maximum frames to process."""
        return self._max_frames

    @property
    def frame_count(self) -> int:
        """Current frame count during iteration."""
        return self._frame_count


class VideoWriter:
    """
    Context manager for writing video files with standard settings.

    Usage:
        with VideoWriter("output.avi", fps=30, size=(1920, 1080)) as writer:
            writer.write(frame)
    """

    # XVID + AVI works reliably across platforms
    DEFAULT_CODEC = "XVID"
    DEFAULT_EXT = ".avi"

    def __init__(
        self,
        path: str,
        fps: int,
        size: Tuple[int, int],
        codec: str = DEFAULT_CODEC,
    ):
        """
        Args:
            path: Output path (extension will be normalized to .avi for XVID)
            fps: Frame rate
            size: (width, height)
            codec: FourCC codec (default: XVID)
        """
        # Ensure .avi extension for XVID codec
        self.path = Path(path)
        if codec == "XVID" and self.path.suffix.lower() != ".avi":
            self.path = self.path.with_suffix(".avi")
        self.path = str(self.path)

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

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._writer:
            self._writer.release()

    def write(self, frame: np.ndarray) -> None:
        """Write a frame (BGR format)."""
        self._writer.write(frame)
        self._frame_count += 1

    @property
    def frame_count(self) -> int:
        """Number of frames written."""
        return self._frame_count


class ProgressTracker:
    """
    Track processing progress with FPS calculation.

    Usage:
        progress = ProgressTracker(total_frames=300, log_interval=30)
        for frame in frames:
            progress.update()
            if progress.should_log():
                print(progress.status())
        print(progress.summary())
    """

    def __init__(self, total_frames: int, log_interval: int = 30):
        """
        Args:
            total_frames: Total frames to process
            log_interval: Frames between log messages
        """
        self.total_frames = total_frames
        self.log_interval = log_interval
        self._start_time: float = time.time()
        self._frame_count: int = 0
        self._last_log_frame: int = 0

    def update(self, n: int = 1) -> None:
        """Update frame count."""
        self._frame_count += n

    def should_log(self) -> bool:
        """Check if it's time to log progress."""
        if self._frame_count - self._last_log_frame >= self.log_interval:
            self._last_log_frame = self._frame_count
            return True
        return False

    @property
    def fps(self) -> float:
        """Current average FPS."""
        elapsed = time.time() - self._start_time
        return self._frame_count / elapsed if elapsed > 0 else 0

    @property
    def percent(self) -> float:
        """Completion percentage."""
        return 100 * self._frame_count / self.total_frames if self.total_frames > 0 else 0

    @property
    def frame_count(self) -> int:
        """Current frame count."""
        return self._frame_count

    def status(self) -> str:
        """Status string for logging."""
        return f"[{self.percent:5.1f}%] Frame {self._frame_count}/{self.total_frames} | {self.fps:.1f} FPS"

    def summary(self) -> str:
        """Final summary string."""
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

    Args:
        input_path: Input video path
        prefix: Output filename prefix
        suffix: Output filename suffix
        duration: Duration in seconds (added to filename if provided)

    Returns:
        Output path like "output_video_10s_tracked.avi"
    """
    input_file = Path(input_path)
    stem = input_file.stem

    parts = [prefix, stem]
    if duration:
        parts.append(f"{int(duration)}s")
    parts.append(suffix)

    output_name = "_".join(parts) + ".avi"
    return output_name
