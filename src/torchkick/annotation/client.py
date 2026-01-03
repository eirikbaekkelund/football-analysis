"""
CVAT REST API client.

This module provides a client for interacting with CVAT (Computer Vision
Annotation Tool) servers, supporting project/task management, image/video
upload, and annotation import/export.

Example:
    >>> from torchkick.annotation import CVATClient, Credentials
    >>> 
    >>> creds = Credentials(
    ...     host="https://app.cvat.ai",
    ...     password="your-api-token",
    ...     use_token=True
    ... )
    >>> client = CVATClient(creds)
    >>> 
    >>> # Create project and task
    >>> project = client.create_project("Match Analysis")
    >>> task = client.create_task("Game 1", project_id=project.id)
    >>> client.upload_video_to_task(task.id, "match.mp4")
"""

from __future__ import annotations

import os
import tempfile
import time
import xml.etree.ElementTree as ET
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from torchkick.annotation.labels import FOOTBALL_LABELS
from torchkick.annotation.models import Annotation, Credentials, Project, Task


class CVATClient:
    """
    Client for interacting with CVAT REST API.

    Supports project and task management, image/video upload,
    annotation import/export, and model-assisted labeling.

    Args:
        credentials: CVAT server credentials.

    Raises:
        Exception: If authentication fails.

    Example:
        >>> client = CVATClient(credentials)
        >>> projects = client.list_projects()
        >>> task = client.get_task(123)
    """

    def __init__(self, credentials: Credentials) -> None:
        self.credentials = credentials
        self.session = requests.Session()
        self._token: Optional[str] = None
        self._authenticate()

    def _authenticate(self) -> None:
        """Authenticate with CVAT server."""
        if self.credentials.use_token:
            self.session.headers.update(
                {
                    "Authorization": f"Bearer {self.credentials.password}",
                    "Content-Type": "application/json",
                }
            )
            try:
                response = self.session.get(f"{self.credentials.base_url}/users/self")
                response.raise_for_status()
                user_data = response.json()
                print(f"Authenticated as: {user_data.get('username', 'unknown')}")
            except Exception as e:
                raise Exception(f"Token authentication failed: {e}")
        else:
            self.session.auth = (self.credentials.username, self.credentials.password)
            self.session.headers.update({"Content-Type": "application/json"})

            try:
                response = self.session.get(f"{self.credentials.base_url}/users/self")
                response.raise_for_status()
                user_data = response.json()
                print(f"Authenticated as: {user_data.get('username', 'unknown')}")
            except Exception as e:
                raise Exception(f"Authentication failed: {e}")

    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> requests.Response:
        """
        Make an authenticated request to CVAT API.

        Args:
            method: HTTP method (GET, POST, PUT, PATCH, DELETE).
            endpoint: API endpoint path.
            data: JSON data for request body.
            files: Files for multipart upload.
            params: Query parameters.

        Returns:
            Response object.

        Raises:
            requests.HTTPError: If request fails.
        """
        url = f"{self.credentials.base_url}/{endpoint}"

        headers = {}
        if files:
            headers = {k: v for k, v in self.session.headers.items() if k != "Content-Type"}

        response = self.session.request(
            method,
            url,
            json=data if not files else None,
            data=data if files else None,
            files=files,
            params=params,
            headers=headers if files else None,
        )
        response.raise_for_status()
        return response

    # -------------------------------------------------------------------------
    # Project Management
    # -------------------------------------------------------------------------

    def create_project(
        self,
        name: str,
        labels: Optional[List[Dict[str, Any]]] = None,
    ) -> Project:
        """
        Create a new CVAT project.

        Args:
            name: Project name.
            labels: Label configurations (defaults to football labels).

        Returns:
            Created Project object.

        Example:
            >>> project = client.create_project("Match 2024-01-01")
        """
        labels = labels or FOOTBALL_LABELS

        response = self._request(
            "POST",
            "projects",
            data={"name": name, "labels": labels},
        )
        data = response.json()

        return Project(
            id=data["id"],
            name=data["name"],
            labels=data.get("labels", []),
            created_date=data["created_date"],
            updated_date=data["updated_date"],
            tasks_count=data.get("tasks_count", 0),
            status=data.get("status", ""),
        )

    def get_project(self, project_id: int) -> Project:
        """
        Get project details by ID.

        Args:
            project_id: Project identifier.

        Returns:
            Project object.
        """
        response = self._request("GET", f"projects/{project_id}")
        data = response.json()

        labels = data.get("labels", [])
        if isinstance(labels, dict):
            labels = []

        return Project(
            id=data["id"],
            name=data["name"],
            labels=labels,
            created_date=data.get("created_date"),
            updated_date=data.get("updated_date"),
            tasks_count=data.get("tasks_count", 0),
            status=data.get("status", ""),
        )

    def list_projects(self, search: Optional[str] = None) -> List[Project]:
        """
        List all projects, optionally filtered by search term.

        Args:
            search: Optional search string to filter projects.

        Returns:
            List of Project objects.
        """
        params = {"search": search} if search else None
        response = self._request("GET", "projects", params=params)
        data = response.json()

        projects = []
        for p in data.get("results", []):
            labels = p.get("labels", [])
            if isinstance(labels, dict):
                labels = []

            projects.append(
                Project(
                    id=p["id"],
                    name=p["name"],
                    labels=labels,
                    created_date=p.get("created_date"),
                    updated_date=p.get("updated_date"),
                    tasks_count=p.get("tasks_count", 0),
                    status=p.get("status", ""),
                )
            )
        return projects

    # -------------------------------------------------------------------------
    # Task Management
    # -------------------------------------------------------------------------

    def create_task(
        self,
        name: str,
        project_id: Optional[int] = None,
        labels: Optional[List[Dict[str, Any]]] = None,
        segment_size: int = 500,
        image_quality: int = 70,
    ) -> Task:
        """
        Create a new annotation task.

        Args:
            name: Task name.
            project_id: Parent project ID (inherits labels from project).
            labels: Labels (only needed if no project specified).
            segment_size: Frames per job segment.
            image_quality: JPEG quality for frame extraction.

        Returns:
            Created Task object.

        Example:
            >>> task = client.create_task("First Half", project_id=project.id)
        """
        task_data: Dict[str, Any] = {
            "name": name,
            "segment_size": segment_size,
            "image_quality": image_quality,
        }

        if project_id:
            task_data["project_id"] = project_id
        elif labels:
            task_data["labels"] = labels
        else:
            task_data["labels"] = FOOTBALL_LABELS

        response = self._request("POST", "tasks", data=task_data)
        data = response.json()

        return Task(
            id=data["id"],
            name=data["name"],
            project_id=data.get("project_id"),
            status=data.get("status", ""),
            size=data.get("size", 0),
            mode=data.get("mode", "annotation"),
            created_date=data["created_date"],
            updated_date=data["updated_date"],
        )

    def get_task(self, task_id: int) -> Task:
        """
        Get task details by ID.

        Args:
            task_id: Task identifier.

        Returns:
            Task object.
        """
        response = self._request("GET", f"tasks/{task_id}")
        data = response.json()

        return Task(
            id=data["id"],
            name=data["name"],
            project_id=data.get("project_id"),
            status=data.get("status", ""),
            size=data.get("size", 0),
            mode=data.get("mode", "annotation"),
            created_date=data["created_date"],
            updated_date=data["updated_date"],
        )

    # -------------------------------------------------------------------------
    # Data Upload
    # -------------------------------------------------------------------------

    def upload_images_to_task(
        self,
        task_id: int,
        image_paths: List[str],
        wait_for_completion: bool = True,
    ) -> None:
        """
        Upload images to a task.

        Args:
            task_id: Target task ID.
            image_paths: List of image file paths.
            wait_for_completion: Wait for upload processing to complete.
        """
        files = []
        for i, path in enumerate(image_paths):
            with open(path, "rb") as f:
                files.append((f"client_files[{i}]", (Path(path).name, f.read(), "image/jpeg")))

        self.session.headers.pop("Content-Type", None)

        response = self.session.post(
            f"{self.credentials.base_url}/tasks/{task_id}/data",
            files=files,
            data={"image_quality": 95},
        )
        response.raise_for_status()

        self.session.headers["Content-Type"] = "application/json"

        if wait_for_completion:
            self._wait_for_task_status(task_id, ["completed", "failed"])

    def upload_video_to_task(
        self,
        task_id: int,
        video_path: str,
        frame_step: int = 1,
        start_frame: int = 0,
        stop_frame: Optional[int] = None,
        wait_for_completion: bool = True,
    ) -> None:
        """
        Upload a video to a task.

        Args:
            task_id: Target task ID.
            video_path: Path to video file.
            frame_step: Extract every Nth frame.
            start_frame: First frame to extract (0-indexed).
            stop_frame: Last frame to extract (exclusive), None for all.
            wait_for_completion: Wait for processing to complete.
        """
        with open(video_path, "rb") as f:
            files = {
                "client_files[0]": (Path(video_path).name, f, "video/mp4"),
            }
            data: Dict[str, Any] = {
                "image_quality": 70,
                "frame_filter": f"step={frame_step}",
                "start_frame": start_frame,
            }
            if stop_frame is not None:
                data["stop_frame"] = stop_frame

            self.session.headers.pop("Content-Type", None)

            response = self.session.post(
                f"{self.credentials.base_url}/tasks/{task_id}/data",
                files=files,
                data=data,
            )
            response.raise_for_status()

            self.session.headers["Content-Type"] = "application/json"

        if wait_for_completion:
            self._wait_for_task_status(task_id, ["completed", "failed"])

    def _wait_for_task_status(
        self,
        task_id: int,
        target_statuses: List[str],
        timeout: int = 600,
        poll_interval: int = 5,
    ) -> str:
        """Wait for task to reach a target status."""
        start = time.time()
        while time.time() - start < timeout:
            response = self._request("GET", f"tasks/{task_id}/status")
            status = response.json().get("state", "")
            if status in target_statuses:
                return status
            time.sleep(poll_interval)
        raise TimeoutError(f"Task {task_id} did not reach target status in {timeout}s")

    # -------------------------------------------------------------------------
    # Annotations
    # -------------------------------------------------------------------------

    def upload_annotations(
        self,
        task_id: int,
        annotations: List[Annotation],
        format: str = "CVAT for images 1.1",
    ) -> None:
        """
        Upload annotations to a task.

        Args:
            task_id: Target task ID.
            annotations: List of Annotation objects.
            format: Annotation format string.
        """
        xml_data = self._annotations_to_cvat_xml(annotations)

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            with zipfile.ZipFile(tmp, "w") as zf:
                zf.writestr("annotations.xml", xml_data)
            tmp_path = tmp.name

        try:
            with open(tmp_path, "rb") as f:
                files = {"annotation_file": ("annotations.zip", f, "application/zip")}
                self.session.headers.pop("Content-Type", None)

                response = self.session.put(
                    f"{self.credentials.base_url}/tasks/{task_id}/annotations",
                    files=files,
                    params={"format": format},
                )
                response.raise_for_status()

                self.session.headers["Content-Type"] = "application/json"
        finally:
            os.unlink(tmp_path)

    def download_annotations(
        self,
        task_id: int,
        format: str = "CVAT for images 1.1",
    ) -> List[Annotation]:
        """
        Download annotations from a task.

        Args:
            task_id: Task ID.
            format: Export format.

        Returns:
            List of Annotation objects.
        """
        response = self._request(
            "GET",
            f"tasks/{task_id}/annotations",
            params={"format": format, "action": "download"},
        )

        if "application/zip" in response.headers.get("Content-Type", ""):
            with zipfile.ZipFile(BytesIO(response.content)) as zf:
                xml_content = zf.read("annotations.xml").decode("utf-8")
        else:
            xml_content = response.text

        return self._parse_cvat_xml(xml_content)

    def _annotations_to_cvat_xml(self, annotations: List[Annotation]) -> str:
        """Convert annotations to CVAT XML format."""
        root = ET.Element("annotations")

        version = ET.SubElement(root, "version")
        version.text = "1.1"

        by_frame: Dict[int, List[Annotation]] = {}
        for ann in annotations:
            by_frame.setdefault(ann.frame, []).append(ann)

        for frame_num, frame_anns in sorted(by_frame.items()):
            image_elem = ET.SubElement(root, "image")
            image_elem.set("id", str(frame_num))
            image_elem.set("name", f"frame_{frame_num:08d}.jpg")

            for ann in frame_anns:
                box_elem = ET.SubElement(image_elem, "box")
                box_elem.set("label", ann.label)
                box_elem.set("xtl", f"{ann.xtl:.2f}")
                box_elem.set("ytl", f"{ann.ytl:.2f}")
                box_elem.set("xbr", f"{ann.xbr:.2f}")
                box_elem.set("ybr", f"{ann.ybr:.2f}")
                box_elem.set("occluded", "1" if ann.occluded else "0")

                for attr_name, attr_value in ann.attributes.items():
                    attr_elem = ET.SubElement(box_elem, "attribute")
                    attr_elem.set("name", attr_name)
                    attr_elem.text = str(attr_value)

        return ET.tostring(root, encoding="unicode")

    def _parse_cvat_xml(self, xml_content: str) -> List[Annotation]:
        """Parse CVAT XML format to annotations."""
        root = ET.fromstring(xml_content)
        annotations = []

        for image_elem in root.findall(".//image"):
            frame = int(image_elem.get("id", 0))

            for box_elem in image_elem.findall("box"):
                attributes = {}
                for attr_elem in box_elem.findall("attribute"):
                    attr_name = attr_elem.get("name")
                    if attr_name:
                        attributes[attr_name] = attr_elem.text

                ann = Annotation(
                    frame=frame,
                    label=box_elem.get("label", ""),
                    xtl=float(box_elem.get("xtl", 0)),
                    ytl=float(box_elem.get("ytl", 0)),
                    xbr=float(box_elem.get("xbr", 0)),
                    ybr=float(box_elem.get("ybr", 0)),
                    occluded=box_elem.get("occluded") == "1",
                    attributes=attributes,
                )
                annotations.append(ann)

        return annotations


# Backwards compatibility alias
Client = CVATClient

__all__ = [
    "CVATClient",
    "Client",
]
