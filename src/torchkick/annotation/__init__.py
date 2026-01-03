"""
Annotation module for CVAT integration and dataset management.

This module provides tools for working with CVAT annotation platform,
including project/task management, video ingestion, pre-labeling with
ML models, and dataset export.

Submodules:
    - models: Pydantic models for annotations and CVAT entities
    - labels: Label constants for football and pitch line annotations
    - client: CVAT REST API client
    - ingestion: Video ingestion and frame extraction
    - dataset: Dataset management and format conversion
    - exporters: Annotation export to training formats
    - prelabel: Model-assisted pre-labeling pipeline

Example:
    >>> from torchkick.annotation import (
    ...     CVATClient,
    ...     Credentials,
    ...     VideoIngestionPipeline,
    ...     process_video_for_cvat,
    ... )
    >>> 
    >>> # Connect to CVAT
    >>> creds = Credentials(host="https://app.cvat.ai", password="token")
    >>> client = CVATClient(creds)
    >>> 
    >>> # Create project and task
    >>> project = client.create_project("Football Tracking")
    >>> task = client.create_task("Match 1", project_id=project.id)
    >>> 
    >>> # Upload video
    >>> client.upload_video_to_task(task.id, "match.mp4")
"""

from torchkick.annotation.models import (
    Annotation,
    Credentials,
    Project,
    Task,
    TrackAnnotation,
)
from torchkick.annotation.labels import (
    FOOTBALL_LABELS,
    LINE_INDEX_TO_NAME,
    PITCH_LINE_LABELS,
    REID_LABELS,
    YOLO_LABEL_MAP,
)
from torchkick.annotation.client import CVATClient, Client
from torchkick.annotation.ingestion import (
    FrameExtractionConfig,
    VideoIngestionPipeline,
    VideoMetadata,
    extract_keyframes,
)
from torchkick.annotation.dataset import (
    ActiveLearningSelector,
    DatasetInfo,
    DatasetManager,
)
from torchkick.annotation.exporters import (
    AnnotationExporter,
    convert_yolo_to_coco,
)
from torchkick.annotation.prelabel import (
    batch_process_videos,
    prelabel_task,
    process_video_for_cvat,
    upload_line_tracks,
    upload_player_tracks,
)

__all__ = [
    # Models
    "Annotation",
    "Credentials",
    "Project",
    "Task",
    "TrackAnnotation",
    # Labels
    "FOOTBALL_LABELS",
    "LINE_INDEX_TO_NAME",
    "PITCH_LINE_LABELS",
    "REID_LABELS",
    "YOLO_LABEL_MAP",
    # Client
    "CVATClient",
    "Client",
    # Ingestion
    "FrameExtractionConfig",
    "VideoIngestionPipeline",
    "VideoMetadata",
    "extract_keyframes",
    # Dataset
    "ActiveLearningSelector",
    "DatasetInfo",
    "DatasetManager",
    # Exporters
    "AnnotationExporter",
    "convert_yolo_to_coco",
    # Prelabel
    "batch_process_videos",
    "prelabel_task",
    "process_video_for_cvat",
    "upload_line_tracks",
    "upload_player_tracks",
]
