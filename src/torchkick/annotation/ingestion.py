"""
Video ingestion and frame extraction utilities.

This module provides tools for ingesting video files into the annotation
pipeline, extracting frames at configurable intervals, and preparing
datasets for annotation.

Example:
    >>> from torchkick.annotation import VideoIngestionPipeline
    >>> 
    >>> pipeline = VideoIngestionPipeline(output_dir="./frames")
    >>> metadata = pipeline.ingest_video("match.mp4", fps=2)
    >>> print(f"Extracted {len(metadata.frame_paths)} frames")
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, Field


class VideoMetadata(BaseModel):
    """
    Metadata about an ingested video.

    Stores video properties, extraction parameters, and paths to
    extracted frames for downstream processing.
    """

    video_path: str = Field(..., description="Original video file path")
    video_id: str = Field(..., description="Unique identifier for the video")
    video_name: str = Field(..., description="Video filename without extension")
    duration_seconds: float = Field(..., description="Video duration in seconds")
    fps: float = Field(..., description="Original video frame rate")
    total_frames: int = Field(..., description="Total number of frames")
    width: int = Field(..., description="Frame width in pixels")
    height: int = Field(..., description="Frame height in pixels")
    extraction_fps: float = Field(..., description="Frame extraction rate")
    frame_paths: List[str] = Field(default_factory=list, description="Paths to extracted frames")
    output_dir: str = Field(..., description="Output directory for frames")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    @property
    def num_extracted_frames(self) -> int:
        """Number of extracted frames."""
        return len(self.frame_paths)


class FrameExtractionConfig(BaseModel):
    """
    Configuration for frame extraction.

    Args:
        fps: Target frames per second for extraction.
        max_frames: Maximum number of frames to extract.
        start_time: Start time in seconds.
        end_time: End time in seconds (None for entire video).
        frame_format: Output image format (jpg, png).
        jpeg_quality: JPEG quality (1-100).
        resize: Target resolution (width, height) or None.
    """

    fps: float = Field(default=1.0, description="Target FPS for extraction")
    max_frames: Optional[int] = Field(default=None, description="Max frames to extract")
    start_time: float = Field(default=0.0, description="Start time in seconds")
    end_time: Optional[float] = Field(default=None, description="End time in seconds")
    frame_format: str = Field(default="jpg", description="Output format (jpg, png)")
    jpeg_quality: int = Field(default=95, ge=1, le=100, description="JPEG quality")
    resize: Optional[Tuple[int, int]] = Field(default=None, description="Target (width, height)")


class VideoIngestionPipeline:
    """
    Pipeline for ingesting videos and extracting frames for annotation.

    Handles video loading, frame extraction at configurable rates,
    and metadata tracking.

    Args:
        output_dir: Base directory for extracted frames.
        config: Frame extraction configuration.

    Example:
        >>> pipeline = VideoIngestionPipeline("./data/frames")
        >>> metadata = pipeline.ingest_video("match.mp4", fps=2.0)
        >>> print(f"Extracted {metadata.num_extracted_frames} frames")
    """

    def __init__(
        self,
        output_dir: str,
        config: Optional[FrameExtractionConfig] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.config = config or FrameExtractionConfig()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def ingest_video(
        self,
        video_path: str,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
        video_id: Optional[str] = None,
    ) -> VideoMetadata:
        """
        Ingest a video and extract frames.

        Args:
            video_path: Path to input video file.
            fps: Override extraction FPS from config.
            max_frames: Override max frames from config.
            video_id: Custom video identifier (auto-generated if None).

        Returns:
            VideoMetadata with extraction results.

        Raises:
            FileNotFoundError: If video file doesn't exist.
            RuntimeError: If video cannot be opened.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Generate video ID
        if video_id is None:
            name_part = video_path.stem.replace(" ", "_").lower()[:20]
            hash_part = hashlib.md5(str(video_path.absolute()).encode()).hexdigest()[:8]
            video_id = f"{name_part}_{hash_part}"

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        try:
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / video_fps if video_fps > 0 else 0

            # Determine extraction parameters
            extract_fps = fps or self.config.fps
            extract_max = max_frames or self.config.max_frames

            # Create output directory
            frames_dir = self.output_dir / video_id / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)

            # Extract frames
            frame_paths = self._extract_frames(
                cap=cap,
                output_dir=frames_dir,
                video_fps=video_fps,
                extract_fps=extract_fps,
                max_frames=extract_max,
                start_time=self.config.start_time,
                end_time=self.config.end_time,
            )

            metadata = VideoMetadata(
                video_path=str(video_path.absolute()),
                video_id=video_id,
                video_name=video_path.stem,
                duration_seconds=duration,
                fps=video_fps,
                total_frames=total_frames,
                width=width,
                height=height,
                extraction_fps=extract_fps,
                frame_paths=[str(p) for p in frame_paths],
                output_dir=str(self.output_dir / video_id),
            )

            # Save metadata
            self._save_metadata(metadata)

            return metadata

        finally:
            cap.release()

    def _extract_frames(
        self,
        cap: cv2.VideoCapture,
        output_dir: Path,
        video_fps: float,
        extract_fps: float,
        max_frames: Optional[int],
        start_time: float,
        end_time: Optional[float],
    ) -> List[Path]:
        """Extract frames from video capture."""
        frame_interval = int(video_fps / extract_fps) if extract_fps > 0 else 1
        start_frame = int(start_time * video_fps)
        end_frame = int(end_time * video_fps) if end_time else None

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_paths = []
        frame_idx = start_frame
        extracted = 0

        ext = self.config.frame_format
        quality = self.config.jpeg_quality

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if end_frame and frame_idx >= end_frame:
                break

            if max_frames and extracted >= max_frames:
                break

            if (frame_idx - start_frame) % frame_interval == 0:
                # Resize if needed
                if self.config.resize:
                    frame = cv2.resize(
                        frame,
                        self.config.resize,
                        interpolation=cv2.INTER_AREA,
                    )

                # Save frame
                frame_path = output_dir / f"frame_{frame_idx:08d}.{ext}"

                if ext == "jpg":
                    cv2.imwrite(
                        str(frame_path),
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, quality],
                    )
                else:
                    cv2.imwrite(str(frame_path), frame)

                frame_paths.append(frame_path)
                extracted += 1

            frame_idx += 1

        print(f"Extracted {len(frame_paths)} frames to {output_dir}")
        return frame_paths

    def _save_metadata(self, metadata: VideoMetadata) -> None:
        """Save video metadata to JSON."""
        meta_path = Path(metadata.output_dir) / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata.model_dump(), f, indent=2)

    def load_metadata(self, video_id: str) -> VideoMetadata:
        """
        Load metadata for a previously ingested video.

        Args:
            video_id: Video identifier.

        Returns:
            VideoMetadata object.

        Raises:
            FileNotFoundError: If metadata doesn't exist.
        """
        meta_path = self.output_dir / video_id / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found for video: {video_id}")

        with open(meta_path) as f:
            data = json.load(f)

        return VideoMetadata(**data)

    def list_videos(self) -> List[str]:
        """
        List all ingested video IDs.

        Returns:
            List of video identifiers.
        """
        videos = []
        for path in self.output_dir.iterdir():
            if path.is_dir() and (path / "metadata.json").exists():
                videos.append(path.name)
        return sorted(videos)


def extract_keyframes(
    video_path: str,
    output_dir: str,
    threshold: float = 30.0,
    max_frames: Optional[int] = None,
) -> List[str]:
    """
    Extract keyframes based on scene changes.

    Uses frame difference to detect significant changes and
    extracts representative frames.

    Args:
        video_path: Path to input video.
        output_dir: Output directory for frames.
        threshold: Scene change threshold (higher = fewer frames).
        max_frames: Maximum frames to extract.

    Returns:
        List of extracted frame paths.

    Example:
        >>> frames = extract_keyframes("match.mp4", "./keyframes", threshold=40)
        >>> print(f"Extracted {len(frames)} keyframes")
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_paths = []
    prev_gray = None
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames and len(frame_paths) >= max_frames:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                mean_diff = np.mean(diff)

                if mean_diff > threshold:
                    frame_path = out_dir / f"keyframe_{frame_idx:08d}.jpg"
                    cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    frame_paths.append(str(frame_path))
            else:
                # Always save first frame
                frame_path = out_dir / f"keyframe_{frame_idx:08d}.jpg"
                cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                frame_paths.append(str(frame_path))

            prev_gray = gray
            frame_idx += 1

    finally:
        cap.release()

    print(f"Extracted {len(frame_paths)} keyframes from {frame_idx} total frames")
    return frame_paths


__all__ = [
    "VideoMetadata",
    "FrameExtractionConfig",
    "VideoIngestionPipeline",
    "extract_keyframes",
]
