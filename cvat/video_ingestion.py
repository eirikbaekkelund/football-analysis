import cv2
import json
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Generator
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from pydantic import BaseModel, Field


class VideoMetadata(BaseModel):
    """Metadata for an ingested video."""

    video_id: str
    source_path: str
    resolution: Tuple[int, int]
    fps: float
    total_frames: int
    duration_seconds: float
    codec: str
    bitrate: Optional[int] = None
    recording_date: Optional[str] = None
    venue: Optional[str] = None
    match_info: Optional[Dict] = None
    quality_score: Optional[float] = None
    ingestion_timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class FrameExtractionConfig(BaseModel):
    """Configuration for frame extraction."""

    frame_extraction_rate: int = 1  # 1 = all frames
    start_time: float = 0.0
    end_time: Optional[float] = None
    min_quality_score: float = 0.3  # blur/quality threshold
    max_frames: Optional[int] = None
    output_format: str = "jpg"
    jpeg_quality: int = 95
    resize: Optional[Tuple[int, int]] = None
    skip_similar_frames: bool = True
    similarity_threshold: float = 0.95


class VideoIngestionPipeline:
    """
    Pipeline for ingesting videos from various sources and preparing them for annotation.

    Supports Veo cameras, broadcast videos, and other sources.
    Extracts frames with quality filtering and prepares them for CVAT upload.
    """

    def __init__(
        self,
        output_dir: str,
        metadata_file: str = "video_metadata.json",
    ):
        """
        Initialize the video ingestion pipeline.

        Args:
            output_dir: Base directory for extracted frames and metadata
            metadata_file: Name of the metadata JSON file
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.output_dir / metadata_file
        self.metadata_cache: Dict[str, VideoMetadata] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load existing metadata from file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                data = json.load(f)
                for vid_id, meta in data.items():
                    self.metadata_cache[vid_id] = VideoMetadata(**meta)

    def _save_metadata(self) -> None:
        """Save metadata to file."""
        data = {vid_id: meta.model_dump_json() for vid_id, meta in self.metadata_cache.items()}
        with open(self.metadata_file, "w") as f:
            json.dump(data, f, indent=2)

    def _generate_video_id(self, video_path: str) -> str:
        """Generate a unique ID for a video based on path and file hash."""
        with open(video_path, "rb") as f:
            # NOTE: 1mb should be sufficient for uniqueness
            file_hash = hashlib.md5(f.read(1024 * 1024)).hexdigest()[:8]
        filename = Path(video_path).stem
        return f"{filename}_{file_hash}"

    def _compute_frame_quality(self, frame: np.ndarray) -> float:
        """
        Compute a quality score for a frame (0-1).
        Uses Laplacian variance for blur detection and histogram analysis.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Laplacian variance (blur detection)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(laplacian_var / 500.0, 1.0)  # Normalize

        # Histogram spread (contrast)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        non_zero_bins = np.sum(hist > 0.001)
        contrast_score = non_zero_bins / 256.0

        # Combined score
        return 0.7 * blur_score + 0.3 * contrast_score

    def _compute_frame_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Compute structural similarity between two frames."""
        # Resize for faster comparison
        size = (128, 72)
        f1 = cv2.resize(frame1, size)
        f2 = cv2.resize(frame2, size)

        # Convert to grayscale
        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Normalized cross-correlation
        g1_norm = (g1 - g1.mean()) / (g1.std() + 1e-6)
        g2_norm = (g2 - g2.mean()) / (g2.std() + 1e-6)

        correlation = np.mean(g1_norm * g2_norm)
        return max(0, min(1, correlation))

    def analyze_video(
        self,
        video_path: str,
        source_type: Optional[str] = None,
        match_info: Optional[Dict] = None,
    ) -> VideoMetadata:
        """
        Analyze a video and extract metadata.

        Args:
            video_path: Path to the video file
            source_type: Optional source type override
            match_info: Optional match information (teams, date, etc.)

        Returns:
            VideoMetadata object with video information
        """
        video_path = str(Path(video_path).resolve())
        video_id = self._generate_video_id(video_path)

        cap = cv2.VideoCapture(video_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        # Get codec
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

        # Sample frames for quality assessment
        quality_samples = []
        sample_interval = max(1, total_frames // 10)
        for i in range(0, min(total_frames, 10 * sample_interval), sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                quality_samples.append(self._compute_frame_quality(frame))

        cap.release()

        avg_quality = np.mean(quality_samples) if quality_samples else 0.5

        metadata = VideoMetadata(
            video_id=video_id,
            source_path=video_path,
            resolution=(width, height),
            fps=fps,
            total_frames=total_frames,
            duration_seconds=duration,
            codec=codec,
            match_info=match_info,
            quality_score=avg_quality,
        )

        self.metadata_cache[video_id] = metadata
        self._save_metadata()

        return metadata

    def extract_frames(
        self,
        video_path: str,
        config: Optional[FrameExtractionConfig] = None,
        progress_callback: Optional[callable] = None,
    ) -> Generator[Tuple[int, str, np.ndarray], None, None]:
        """
        Extract frames from a video with quality filtering.

        Args:
            video_path: Path to the video file
            config: Frame extraction configuration
            progress_callback: Optional callback for progress updates

        Yields:
            Tuples of (frame_number, output_path, frame_array)
        """
        config = config or FrameExtractionConfig()
        video_path = str(Path(video_path).resolve())
        video_id = self._generate_video_id(video_path)

        # Create output directory for this video
        frames_dir = self.output_dir / video_id / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame range
        start_frame = int(config.start_time * fps)
        end_frame = int(config.end_time * fps) if config.end_time else total_frames

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_count = 0
        extracted_count = 0
        prev_frame = None

        pbar = tqdm(total=(end_frame - start_frame) // config.frame_extraction_rate, desc="Extracting frames")

        while True:
            ret, frame = cap.read()
            current_frame = start_frame + frame_count

            if not ret or current_frame >= end_frame:
                break

            if config.max_frames and extracted_count >= config.max_frames:
                break

            if frame_count % config.frame_extraction_rate != 0:
                frame_count += 1
                continue

            quality = self._compute_frame_quality(frame)
            if quality < config.min_quality_score:
                frame_count += 1
                continue

            if config.skip_similar_frames and prev_frame is not None:
                similarity = self._compute_frame_similarity(frame, prev_frame)
                if similarity > config.similarity_threshold:
                    frame_count += 1
                    continue

            if config.resize:
                frame = cv2.resize(frame, config.resize)

            frame_filename = f"frame_{current_frame:08d}.{config.output_format}"
            frame_path = frames_dir / frame_filename

            if config.output_format == "jpg":
                cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality])
            else:
                cv2.imwrite(str(frame_path), frame)

            prev_frame = frame.copy()
            extracted_count += 1
            pbar.update(1)

            if progress_callback:
                progress_callback(current_frame, total_frames)

            yield current_frame, str(frame_path), frame

            frame_count += 1

        pbar.close()
        cap.release()

        print(f"Extracted {extracted_count} frames from {video_path}")

    def extract_frames_batch(
        self,
        video_paths: List[str],
        config: Optional[FrameExtractionConfig] = None,
        num_workers: int = 4,
    ) -> Dict[str, List[str]]:
        """
        Extract frames from multiple videos in parallel.

        Args:
            video_paths: List of video file paths
            config: Frame extraction configuration
            num_workers: Number of parallel workers

        Returns:
            Dictionary mapping video IDs to lists of extracted frame paths
        """
        results = {}

        def process_video(video_path: str) -> Tuple[str, List[str]]:
            video_id = self._generate_video_id(video_path)
            frame_paths = [path for _, path, _ in self.extract_frames(video_path, config)]
            return video_id, frame_paths

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_video, vp) for vp in video_paths]
            for future in futures:
                video_id, frame_paths = future.result()
                results[video_id] = frame_paths

        return results
