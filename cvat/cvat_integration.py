import os
import time
import zipfile
import tempfile
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from pydantic import BaseModel, Field
from io import BytesIO


class Credentials(BaseModel):
    host: str
    username: str
    password: str
    organization: Optional[str] = None
    use_token: bool = False

    @property
    def base_url(self) -> str:
        return f"{self.host}/api"


class Project(BaseModel):
    """project representation."""

    model_config = {"extra": "ignore"}

    id: int
    name: str
    labels: Optional[List[Dict]] = None  # Can be list or URL reference
    created_date: Optional[str] = None
    updated_date: Optional[str] = None
    tasks_count: Optional[int] = 0
    status: Optional[str] = None


class Task(BaseModel):
    """task representation."""

    model_config = {"extra": "ignore"}

    id: int
    name: str
    project_id: Optional[int] = None
    status: Optional[str] = None
    num_frames: Optional[int] = None
    mode: Optional[str] = None
    created_date: Optional[str] = None
    updated_date: Optional[str] = None
    size: Optional[int] = None  # Alternative to num_frames


class Annotation(BaseModel):
    """Single annotation (bounding box with attributes)."""

    frame: int
    label: str
    xtl: float  # x top-left
    ytl: float  # y top-left
    xbr: float  # x bottom-right
    ybr: float  # y bottom-right
    occluded: bool = False
    attributes: Dict[str, Any] = Field(default_factory=dict)
    track_id: Optional[int] = None


FOOTBALL_LABELS = [
    {
        "name": "player",
        "color": "#ff0000",
        "attributes": [
            {"name": "team", "mutable": True, "input_type": "select", "values": ["team_a", "team_b", "unknown"]},
            {"name": "jersey_number", "mutable": True, "input_type": "number", "values": ["0", "99"]},
            {"name": "role", "mutable": False, "input_type": "select", "values": ["outfield", "goalkeeper"]},
        ],
    },
    {
        "name": "referee",
        "color": "#ffff00",
        "attributes": [
            {"name": "role", "mutable": False, "input_type": "select", "values": ["main", "assistant", "fourth"]},
        ],
    },
    {
        "name": "goalkeeper",
        "color": "#00ff00",
        "attributes": [
            {"name": "team", "mutable": True, "input_type": "select", "values": ["team_a", "team_b"]},
            {"name": "jersey_number", "mutable": True, "input_type": "number", "values": ["0", "99"]},
        ],
    },
]

PITCH_LINE_LABELS = [
    {"name": "Side line top", "color": "#00ff00", "type": "polyline", "attributes": []},
    {"name": "Side line bottom", "color": "#00ff00", "type": "polyline", "attributes": []},
    {"name": "Side line left", "color": "#00ff00", "type": "polyline", "attributes": []},
    {"name": "Side line right", "color": "#00ff00", "type": "polyline", "attributes": []},
    {"name": "Middle line", "color": "#ffff00", "type": "polyline", "attributes": []},
    {"name": "Big rect. left main", "color": "#ff00ff", "type": "polyline", "attributes": []},
    {"name": "Big rect. left top", "color": "#ff00ff", "type": "polyline", "attributes": []},
    {"name": "Big rect. left bottom", "color": "#ff00ff", "type": "polyline", "attributes": []},
    {"name": "Big rect. right main", "color": "#ff00ff", "type": "polyline", "attributes": []},
    {"name": "Big rect. right top", "color": "#ff00ff", "type": "polyline", "attributes": []},
    {"name": "Big rect. right bottom", "color": "#ff00ff", "type": "polyline", "attributes": []},
    {"name": "Small rect. left main", "color": "#00ffff", "type": "polyline", "attributes": []},
    {"name": "Small rect. left top", "color": "#00ffff", "type": "polyline", "attributes": []},
    {"name": "Small rect. left bottom", "color": "#00ffff", "type": "polyline", "attributes": []},
    {"name": "Small rect. right main", "color": "#00ffff", "type": "polyline", "attributes": []},
    {"name": "Small rect. right top", "color": "#00ffff", "type": "polyline", "attributes": []},
    {"name": "Small rect. right bottom", "color": "#00ffff", "type": "polyline", "attributes": []},
    {"name": "Circle central", "color": "#ffffff", "type": "ellipse", "attributes": []},
    {"name": "Circle left", "color": "#ffffff", "type": "ellipse", "attributes": []},
    {"name": "Circle right", "color": "#ffffff", "type": "ellipse", "attributes": []},
    {"name": "Goal left crossbar", "color": "#ff0000", "type": "polyline", "attributes": []},
    {"name": "Goal left post left", "color": "#ff0000", "type": "polyline", "attributes": []},
    {"name": "Goal left post right", "color": "#ff0000", "type": "polyline", "attributes": []},
    {"name": "Goal right crossbar", "color": "#ff0000", "type": "polyline", "attributes": []},
    {"name": "Goal right post left", "color": "#ff0000", "type": "polyline", "attributes": []},
    {"name": "Goal right post right", "color": "#ff0000", "type": "polyline", "attributes": []},
]

# Re-identification specific labels
REID_LABELS = [
    {
        "name": "person",
        "color": "#ff6600",
        "attributes": [
            {"name": "person_id", "mutable": False, "input_type": "text", "values": [""]},
            {
                "name": "category",
                "mutable": False,
                "input_type": "select",
                "values": ["player", "referee", "staff", "other"],
            },
            {
                "name": "team",
                "mutable": True,
                "input_type": "select",
                "values": ["team_a", "team_b", "neutral", "unknown"],
            },
            {"name": "jersey_color", "mutable": True, "input_type": "text", "values": [""]},
            {"name": "shorts_color", "mutable": True, "input_type": "text", "values": [""]},
        ],
    },
]


class Client:
    """
    Client for interacting with  REST API.

    Supports:
    - Project and task management
    - Image/video upload
    - Annotation import/export
    - Model-assisted labeling
    """

    def __init__(self, credentials: Credentials):
        """
        Initialize  client.

        Args:
            credentials:  server credentials
        """
        self.credentials = credentials
        self.session = requests.Session()
        self._token: Optional[str] = None
        self._authenticate()

    def _authenticate(self) -> None:
        """Authenticate with  server."""
        #  Cloud and newer versions use basic auth or session auth
        if self.credentials.use_token:
            # Token-based auth -  Cloud uses "Bearer" prefix
            self.session.headers.update(
                {
                    "Authorization": f"Bearer {self.credentials.password}",
                    "Content-Type": "application/json",
                }
            )
            # Verify authentication
            try:
                response = self.session.get(f"{self.credentials.base_url}/users/self")
                response.raise_for_status()
                user_data = response.json()
                print(f"   Authenticated as: {user_data.get('username', 'unknown')}")
            except Exception as e:
                raise Exception(f"Token authentication failed: {e}")
        else:
            self.session.auth = (self.credentials.username, self.credentials.password)
            self.session.headers.update(
                {
                    "Content-Type": "application/json",
                }
            )

            try:
                response = self.session.get(f"{self.credentials.base_url}/users/self")
                response.raise_for_status()
                user_data = response.json()
                print(f"   Authenticated as: {user_data.get('username', 'unknown')}")
            except Exception as e:
                raise Exception(f"Authentication failed: {e}")

    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        files: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> requests.Response:
        """Make an authenticated request to  API."""
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

    def create_project(
        self,
        name: str,
        labels: Optional[List[Dict]] = None,
    ) -> Project:
        """
        Create a new  project.

        Args:
            name: Project name
            labels: List of label configurations (defaults to football labels)
            description: Project description

        Returns:
            Project object
        """
        labels = labels or FOOTBALL_LABELS

        response = self._request(
            "POST",
            "projects",
            data={
                "name": name,
                "labels": labels,
            },
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
        """Get project details by ID."""
        response = self._request("GET", f"projects/{project_id}")
        data = response.json()

        # Handle labels - can be a list or a URL reference dict
        labels = data.get("labels", [])
        if isinstance(labels, dict):
            labels = []  # URL reference, ignore for now

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
        """List all projects, optionally filtered by search term."""
        params = {"search": search} if search else None
        response = self._request("GET", "projects", params=params)
        data = response.json()

        projects = []
        for p in data.get("results", []):
            # Handle labels - can be a list or a URL reference dict
            labels = p.get("labels", [])
            if isinstance(labels, dict):
                labels = []  # URL reference, ignore for now

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

    def create_task(
        self,
        name: str,
        project_id: Optional[int] = None,
        labels: Optional[List[Dict]] = None,
        segment_size: int = 500,
        image_quality: int = 70,
    ) -> Task:
        """
        Create a new annotation task.

        Args:
            name: Task name
            project_id: Parent project ID
            labels: Labels (only needed if no project)
            segment_size: Frames per job segment
            image_quality: JPEG quality for frames

        Returns:
            Task object
        """
        task_data = {
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

    def upload_images_to_task(
        self,
        task_id: int,
        image_paths: List[str],
        wait_for_completion: bool = True,
    ) -> None:
        """
        Upload images to a task.

        Args:
            task_id: Target task ID
            image_paths: List of image file paths
            wait_for_completion: Wait for upload processing to complete
        """
        files = []
        for i, path in enumerate(image_paths):
            with open(path, "rb") as f:
                files.append((f"client_files[{i}]", (Path(path).name, f.read(), "image/jpeg")))

        # Remove JSON content-type for file upload
        self.session.headers.pop("Content-Type", None)

        response = self.session.post(
            f"{self.credentials.base_url}/tasks/{task_id}/data",
            files=files,
            data={"image_quality": 95},
        )
        response.raise_for_status()

        # Restore content type
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
            task_id: Target task ID
            video_path: Path to video file
            frame_step: Extract every Nth frame
            start_frame: First frame to extract (0-indexed)
            stop_frame: Last frame to extract (exclusive), None for all
            wait_for_completion: Wait for processing to complete
        """
        with open(video_path, "rb") as f:
            files = {
                "client_files[0]": (Path(video_path).name, f, "video/mp4"),
            }
            data = {
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

    def get_task(self, task_id: int) -> Task:
        """Get task details by ID."""
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

    def upload_annotations(
        self,
        task_id: int,
        annotations: List[Annotation],
        format: str = " for images 1.1",
    ) -> None:
        """
        Upload annotations to a task.

        Args:
            task_id: Target task ID
            annotations: List of annotations
            format: Annotation format
        """
        xml_data = self._annotations_to__xml(annotations)

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
        format: str = " for images 1.1",
    ) -> List[Annotation]:
        """
        Download annotations from a task.

        Args:
            task_id: Task ID
            format: Export format

        Returns:
            List of Annotation objects
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

        return self._parse__xml(xml_content)

    def _annotations_to__xml(self, annotations: List[Annotation]) -> str:
        """Convert annotations to  XML format."""
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

    def _parse__xml(self, xml_content: str) -> List[Annotation]:
        """Parse  XML format to annotations."""
        root = ET.fromstring(xml_content)
        annotations = []

        for image_elem in root.findall(".//image"):
            frame = int(image_elem.get("id", 0))

            for box_elem in image_elem.findall("box"):
                attributes = {}
                for attr_elem in box_elem.findall("attribute"):
                    attributes[attr_elem.get("name")] = attr_elem.text

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


class AnnotationExporter:
    """
    Export annotations from  to various training formats.

    Supports:
    - YOLO format (for object detection)
    - MOT format (for tracking)
    - COCO format (for detection/segmentation)
    - ReID format (for re-identification training)
    """

    YOLO_LABEL_MAP = {
        "player": 0,
        "referee": 1,
        "goalkeeper": 2,
        "ball": 3,
    }

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_to_yolo(
        self,
        annotations: List[Annotation],
        images_dir: str,
        image_width: int,
        image_height: int,
        label_map: Optional[Dict[str, int]] = None,
    ) -> str:
        """
        Export annotations to YOLO format.

        Args:
            annotations: List of  annotations
            images_dir: Directory containing images
            image_width: Image width for normalization
            image_height: Image height for normalization
            label_map: Custom label to class ID mapping

        Returns:
            Path to output directory
        """
        label_map = label_map or self.YOLO_LABEL_MAP

        yolo_dir = self.output_dir / "yolo"
        labels_dir = yolo_dir / "labels" / "train"
        images_out_dir = yolo_dir / "images" / "train"

        labels_dir.mkdir(parents=True, exist_ok=True)
        images_out_dir.mkdir(parents=True, exist_ok=True)

        by_frame: Dict[int, List[Annotation]] = {}
        for ann in annotations:
            by_frame.setdefault(ann.frame, []).append(ann)

        for frame_num, frame_anns in by_frame.items():
            label_lines = []

            for ann in frame_anns:
                if ann.label not in label_map:
                    continue

                class_id = label_map[ann.label]

                x_center = (ann.xtl + ann.xbr) / 2 / image_width
                y_center = (ann.ytl + ann.ybr) / 2 / image_height
                width = (ann.xbr - ann.xtl) / image_width
                height = (ann.ybr - ann.ytl) / image_height

                label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            label_path = labels_dir / f"frame_{frame_num:08d}.txt"
            with open(label_path, "w") as f:
                f.write("\n".join(label_lines))

        yaml_content = f"""
            path: {yolo_dir.resolve()}
            train: images/train
            val: images/train
            names:
            """.strip()
        for label, class_id in sorted(label_map.items(), key=lambda x: x[1]):
            yaml_content += f"  {class_id}: {label}\n"

        yaml_path = yolo_dir / "dataset.yaml"
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        return str(yolo_dir)

    def export_to_mot(
        self,
        annotations: List[Annotation],
        output_name: str = "gt",
    ) -> str:
        """
        Export annotations to MOT (Multi-Object Tracking) format.

        MOT format: frame,id,bb_left,bb_top,bb_width,bb_height,conf,class,visibility

        Args:
            annotations: List of  annotations
            output_name: Output filename (without extension)

        Returns:
            Path to output file
        """
        mot_dir = self.output_dir / "mot"
        mot_dir.mkdir(parents=True, exist_ok=True)

        lines = []
        for ann in annotations:
            track_id = ann.track_id or -1

            frame = ann.frame + 1
            bb_left = ann.xtl
            bb_top = ann.ytl
            bb_width = ann.xbr - ann.xtl
            bb_height = ann.ybr - ann.ytl
            conf = 1.0
            class_id = self.YOLO_LABEL_MAP.get(ann.label, -1)
            visibility = 0.0 if ann.occluded else 1.0

            lines.append(
                f"{frame},{track_id},{bb_left:.2f},{bb_top:.2f},{bb_width:.2f},{bb_height:.2f},{conf},{class_id},{visibility}"
            )

        output_path = mot_dir / f"{output_name}.txt"
        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        return str(output_path)

    def export_to_reid(
        self,
        annotations: List[Annotation],
        images_dir: str,
        crop_size: Tuple[int, int] = (128, 256),
    ) -> str:
        """
        Export annotations to ReID format (cropped person images organized by ID).

        Args:
            annotations: List of  annotations
            images_dir: Directory containing source images
            crop_size: Output crop size (width, height)

        Returns:
            Path to output directory
        """
        import cv2

        reid_dir = self.output_dir / "reid"
        reid_dir.mkdir(parents=True, exist_ok=True)

        images_dir = Path(images_dir)

        for ann in annotations:
            person_id = ann.attributes.get("person_id", "unknown")
            if not person_id or person_id == "unknown":
                continue

            person_dir = reid_dir / str(person_id)
            person_dir.mkdir(exist_ok=True)

            frame_path = images_dir / f"frame_{ann.frame:08d}.jpg"
            if not frame_path.exists():
                continue

            img = cv2.imread(str(frame_path))
            if img is None:
                continue

            x1, y1 = int(ann.xtl), int(ann.ytl)
            x2, y2 = int(ann.xbr), int(ann.ybr)

            # Ensure valid crop
            h, w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = img[y1:y2, x1:x2]
            crop = cv2.resize(crop, crop_size)

            # Save with unique filename
            crop_name = f"{ann.frame:08d}_{int(ann.xtl):04d}_{int(ann.ytl):04d}.jpg"
            cv2.imwrite(str(person_dir / crop_name), crop)

        return str(reid_dir)


class ModelAssistedLabeling:
    """
    Model-assisted labeling for  integration.

    Uses trained models to generate pre-annotations that annotators
    can review and correct, speeding up the annotation process.
    """

    def __init__(
        self,
        detection_model_path: Optional[str] = None,
        reid_model_path: Optional[str] = None,
    ):
        """
        Initialize with model paths.

        Args:
            detection_model_path: Path to YOLO detection model
            reid_model_path: Path to ReID model
        """
        self.detection_model = None
        self.reid_model = None

        if detection_model_path:
            self._load_detection_model(detection_model_path)
        if reid_model_path:
            self._load_reid_model(reid_model_path)

    def _load_detection_model(self, model_path: str) -> None:
        """Load YOLO detection model."""
        try:
            from ultralytics import YOLO

            self.detection_model = YOLO(model_path)
        except ImportError:
            print("Warning: ultralytics not installed, detection model not loaded")

    def _load_reid_model(self, model_path: str) -> None:
        """Load ReID model."""
        # Will be implemented with the ReID model
        pass

    def generate_predictions(
        self,
        image_paths: List[str],
        confidence_threshold: float = 0.5,
    ) -> List[Annotation]:
        """
        Generate predictions for a list of images.

        Args:
            image_paths: List of image file paths
            confidence_threshold: Minimum confidence for predictions

        Returns:
            List of Annotation objects
        """
        if self.detection_model is None:
            raise ValueError("No detection model loaded")

        label_map = {0: "player", 1: "referee", 2: "goalkeeper", 3: "ball"}
        annotations = []

        for i, path in enumerate(image_paths):
            results = self.detection_model(path, verbose=False)[0]

            for box in results.boxes:
                if box.conf[0] < confidence_threshold:
                    continue

                class_id = int(box.cls[0])
                xyxy = box.xyxy[0].cpu().numpy()

                ann = Annotation(
                    frame=i,
                    label=label_map.get(class_id, "unknown"),
                    xtl=float(xyxy[0]),
                    ytl=float(xyxy[1]),
                    xbr=float(xyxy[2]),
                    ybr=float(xyxy[3]),
                )
                annotations.append(ann)

        return annotations

    def upload_predictions_to_(
        self,
        _client: Client,
        task_id: int,
        image_paths: List[str],
        confidence_threshold: float = 0.5,
    ) -> int:
        """
        Generate and upload predictions to a  task.

        Args:
            _client:  client instance
            task_id: Target task ID
            image_paths: List of image paths
            confidence_threshold: Minimum confidence

        Returns:
            Number of annotations uploaded
        """
        annotations = self.generate_predictions(image_paths, confidence_threshold)
        _client.upload_annotations(task_id, annotations)
        return len(annotations)
