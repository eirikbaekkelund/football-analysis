"""
Annotation export and format conversion utilities.

This module provides tools for exporting annotations from CVAT to various
formats (YOLO, COCO, Pascal VOC) for training and evaluation.

Example:
    >>> from torchkick.annotation import AnnotationExporter
    >>> 
    >>> exporter = AnnotationExporter(client)
    >>> exporter.export_yolo(task_id=123, output_dir="./dataset")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from torchkick.annotation.labels import YOLO_LABEL_MAP
from torchkick.annotation.models import Annotation


class AnnotationExporter:
    """
    Export CVAT annotations to various training formats.

    Supports YOLO, COCO, and Pascal VOC formats for detector training.

    Args:
        client: Authenticated CVATClient instance.

    Example:
        >>> exporter = AnnotationExporter(client)
        >>> exporter.export_yolo(task_id=123, output_dir="dataset/")
    """

    def __init__(self, client: Any) -> None:
        self.client = client

    def export_yolo(
        self,
        task_id: int,
        output_dir: str,
        label_map: Optional[Dict[str, int]] = None,
        train_split: float = 0.8,
    ) -> Dict[str, Any]:
        """
        Export task annotations to YOLO format.

        Creates the following structure:
        ```
        output_dir/
        ├── data.yaml
        ├── images/
        │   ├── train/
        │   └── val/
        └── labels/
            ├── train/
            └── val/
        ```

        Args:
            task_id: CVAT task ID.
            output_dir: Output directory path.
            label_map: Custom label to class ID mapping.
            train_split: Fraction of data for training.

        Returns:
            Dict with export statistics.
        """
        label_map = label_map or YOLO_LABEL_MAP

        out = Path(output_dir)
        (out / "images" / "train").mkdir(parents=True, exist_ok=True)
        (out / "images" / "val").mkdir(parents=True, exist_ok=True)
        (out / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (out / "labels" / "val").mkdir(parents=True, exist_ok=True)

        # Download annotations
        annotations = self.client.download_annotations(task_id)

        # Group by frame
        by_frame: Dict[int, List[Annotation]] = {}
        for ann in annotations:
            by_frame.setdefault(ann.frame, []).append(ann)

        # Split frames
        frames = sorted(by_frame.keys())
        split_idx = int(len(frames) * train_split)
        train_frames = set(frames[:split_idx])

        stats = {"total_frames": len(frames), "train_frames": len(train_frames)}
        stats["val_frames"] = len(frames) - len(train_frames)

        # Write label files
        for frame_num, frame_anns in by_frame.items():
            split = "train" if frame_num in train_frames else "val"
            label_file = out / "labels" / split / f"frame_{frame_num:08d}.txt"

            lines = []
            for ann in frame_anns:
                if ann.label not in label_map:
                    continue

                class_id = label_map[ann.label]
                x_center = (ann.xtl + ann.xbr) / 2
                y_center = (ann.ytl + ann.ybr) / 2
                width = ann.xbr - ann.xtl
                height = ann.ybr - ann.ytl

                # Note: YOLO expects normalized coordinates
                # Assuming 1920x1080 for now - should get from task metadata
                lines.append(
                    f"{class_id} {x_center/1920:.6f} {y_center/1080:.6f} " f"{width/1920:.6f} {height/1080:.6f}"
                )

            label_file.write_text("\n".join(lines))

        # Write data.yaml
        class_names = {v: k for k, v in label_map.items()}
        data_yaml = {
            "path": str(out.absolute()),
            "train": "images/train",
            "val": "images/val",
            "names": class_names,
        }

        with open(out / "data.yaml", "w") as f:
            import yaml

            yaml.dump(data_yaml, f)

        return stats

    def export_coco(
        self,
        task_id: int,
        output_path: str,
        label_map: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        Export task annotations to COCO JSON format.

        Args:
            task_id: CVAT task ID.
            output_path: Output JSON file path.
            label_map: Custom label to category ID mapping.

        Returns:
            Dict with export statistics.
        """
        label_map = label_map or YOLO_LABEL_MAP
        annotations = self.client.download_annotations(task_id)

        coco = {
            "images": [],
            "annotations": [],
            "categories": [{"id": v, "name": k} for k, v in label_map.items()],
        }

        image_ids: Dict[int, int] = {}
        ann_id = 1

        for ann in annotations:
            # Create image entry if needed
            if ann.frame not in image_ids:
                img_id = len(image_ids) + 1
                image_ids[ann.frame] = img_id
                coco["images"].append(
                    {
                        "id": img_id,
                        "file_name": f"frame_{ann.frame:08d}.jpg",
                        "width": 1920,  # TODO: get from task
                        "height": 1080,
                    }
                )

            if ann.label not in label_map:
                continue

            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_ids[ann.frame],
                    "category_id": label_map[ann.label],
                    "bbox": [ann.xtl, ann.ytl, ann.xbr - ann.xtl, ann.ybr - ann.ytl],
                    "area": (ann.xbr - ann.xtl) * (ann.ybr - ann.ytl),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(coco, f, indent=2)

        return {
            "images": len(coco["images"]),
            "annotations": len(coco["annotations"]),
        }


def convert_yolo_to_coco(
    yolo_dir: str,
    output_path: str,
    image_width: int = 1920,
    image_height: int = 1080,
) -> Dict[str, Any]:
    """
    Convert YOLO format dataset to COCO JSON format.

    Args:
        yolo_dir: Path to YOLO dataset directory.
        output_path: Output JSON file path.
        image_width: Image width for denormalization.
        image_height: Image height for denormalization.

    Returns:
        Dict with conversion statistics.

    Example:
        >>> stats = convert_yolo_to_coco("dataset/", "coco.json")
        >>> print(f"Converted {stats['annotations']} annotations")
    """
    import yaml

    yolo_path = Path(yolo_dir)

    # Load data.yaml
    with open(yolo_path / "data.yaml") as f:
        data_config = yaml.safe_load(f)

    class_names = data_config.get("names", {})

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": int(k) if isinstance(k, (int, str)) else k, "name": v} for k, v in class_names.items()],
    }

    ann_id = 1
    img_id = 0

    for split in ["train", "val"]:
        labels_dir = yolo_path / "labels" / split
        if not labels_dir.exists():
            continue

        for label_file in sorted(labels_dir.glob("*.txt")):
            img_id += 1

            coco["images"].append(
                {
                    "id": img_id,
                    "file_name": label_file.stem + ".jpg",
                    "width": image_width,
                    "height": image_height,
                }
            )

            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    class_id = int(parts[0])
                    x_center = float(parts[1]) * image_width
                    y_center = float(parts[2]) * image_height
                    width = float(parts[3]) * image_width
                    height = float(parts[4]) * image_height

                    x_min = x_center - width / 2
                    y_min = y_center - height / 2

                    coco["annotations"].append(
                        {
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": class_id,
                            "bbox": [x_min, y_min, width, height],
                            "area": width * height,
                            "iscrowd": 0,
                        }
                    )
                    ann_id += 1

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

    return {
        "images": len(coco["images"]),
        "annotations": len(coco["annotations"]),
    }


__all__ = [
    "AnnotationExporter",
    "convert_yolo_to_coco",
]
