"""
Data models for CVAT annotation integration.

This module provides Pydantic models for representing CVAT entities
like credentials, projects, tasks, and annotations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class Credentials(BaseModel):
    """
    CVAT server authentication credentials.

    Supports both basic authentication (username/password) and token-based
    authentication for CVAT Cloud.

    Attributes:
        host: CVAT server URL (e.g., "https://app.cvat.ai").
        username: Username for basic auth.
        password: Password or API token.
        organization: Optional organization slug for CVAT Cloud.
        use_token: If True, use Bearer token authentication.

    Example:
        >>> creds = Credentials(
        ...     host="https://app.cvat.ai",
        ...     username="",
        ...     password="your-api-token",
        ...     use_token=True
        ... )
    """

    host: str
    username: str = ""
    password: str = ""
    organization: Optional[str] = None
    use_token: bool = False

    @property
    def base_url(self) -> str:
        """Base API URL for CVAT server."""
        return f"{self.host}/api"


class Project(BaseModel):
    """
    CVAT project representation.

    Attributes:
        id: Unique project identifier.
        name: Project display name.
        labels: List of label configurations.
        created_date: ISO timestamp of creation.
        updated_date: ISO timestamp of last update.
        tasks_count: Number of tasks in project.
        status: Project status string.
    """

    model_config = {"extra": "ignore"}

    id: int
    name: str
    labels: Optional[List[Dict[str, Any]]] = None
    created_date: Optional[str] = None
    updated_date: Optional[str] = None
    tasks_count: Optional[int] = 0
    status: Optional[str] = None


class Task(BaseModel):
    """
    CVAT annotation task representation.

    Attributes:
        id: Unique task identifier.
        name: Task display name.
        project_id: Parent project ID (if any).
        status: Task status (annotation, validation, completed).
        num_frames: Number of frames in task.
        mode: Task mode (annotation, interpolation).
        created_date: ISO timestamp of creation.
        updated_date: ISO timestamp of last update.
        size: Alternative to num_frames.
    """

    model_config = {"extra": "ignore"}

    id: int
    name: str
    project_id: Optional[int] = None
    status: Optional[str] = None
    num_frames: Optional[int] = None
    mode: Optional[str] = None
    created_date: Optional[str] = None
    updated_date: Optional[str] = None
    size: Optional[int] = None


class Annotation(BaseModel):
    """
    Single bounding box annotation.

    Represents a rectangular annotation with optional attributes
    and track ID for multi-frame tracking.

    Attributes:
        frame: Frame number (0-indexed).
        label: Label name (e.g., "player", "referee").
        xtl: X coordinate of top-left corner.
        ytl: Y coordinate of top-left corner.
        xbr: X coordinate of bottom-right corner.
        ybr: Y coordinate of bottom-right corner.
        occluded: Whether the object is partially hidden.
        attributes: Additional attributes (team, jersey number, etc.).
        track_id: Track identifier for linking across frames.

    Example:
        >>> ann = Annotation(
        ...     frame=0,
        ...     label="player",
        ...     xtl=100.0, ytl=200.0, xbr=150.0, ybr=400.0,
        ...     attributes={"team": "team_a", "jersey_number": "10"}
        ... )
    """

    frame: int
    label: str
    xtl: float
    ytl: float
    xbr: float
    ybr: float
    occluded: bool = False
    attributes: Dict[str, Any] = Field(default_factory=dict)
    track_id: Optional[int] = None


class TrackAnnotation(BaseModel):
    """
    Multi-frame track annotation for a single object.

    Represents a tracked object across multiple frames, with
    consistent identity and attributes.

    Attributes:
        track_id: Unique track identifier.
        label: Object label (player, goalkeeper, referee).
        frames: List of frame numbers where object appears.
        boxes: List of (x1, y1, x2, y2) bounding boxes per frame.
        attributes: Object attributes (team, role, etc.).

    Example:
        >>> track = TrackAnnotation(
        ...     track_id=1,
        ...     label="player",
        ...     frames=[0, 1, 2],
        ...     boxes=[(100, 200, 150, 400), (102, 198, 152, 398), (105, 195, 155, 395)],
        ...     attributes={"team": "team_a"}
        ... )
    """

    track_id: int
    label: str
    frames: List[int]
    boxes: List[Tuple[float, float, float, float]]
    attributes: Dict[str, str] = Field(default_factory=dict)


__all__ = [
    "Credentials",
    "Project",
    "Task",
    "Annotation",
    "TrackAnnotation",
]
