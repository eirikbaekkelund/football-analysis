import cv2
import json
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Literal
from pydantic import BaseModel
from tqdm import tqdm


class DatasetInfo(BaseModel):
    name: str
    version: str
    created_date: str
    source: Literal["soccernet", "veo", "broadcast", "cvat", "mixed"]
    num_images: int
    num_annotations: int
    classes: List[str]
    splits: Dict[str, int]  # train, val, test counts
    format: str  # yolo, coco, mot, etc.
    path: str
    description: str = ""


class DatasetManager:
    def __init__(self, base_dir: str):
        """
        Initialize dataset manager.

        Args:
            base_dir: Base directory for all datasets
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.datasets_file = self.base_dir / "datasets.json"
        self.datasets: Dict[str, DatasetInfo] = {}
        self._load_datasets_index()

    def _load_datasets_index(self) -> None:
        if self.datasets_file.exists():
            with open(self.datasets_file, "r") as f:
                data = json.load(f)
                for name, info in data.items():
                    self.datasets[name] = DatasetInfo(**info)

    def _save_datasets_index(self) -> None:
        """Save datasets index to file."""
        data = {name: info.model_dump_json() for name, info in self.datasets.items()}
        with open(self.datasets_file, "w") as f:
            json.dump(data, f, indent=2)

    def register_dataset(
        self,
        name: str,
        path: str,
        source: str,
        format: str,
        classes: List[str],
        description: str = "",
    ) -> DatasetInfo:
        """
        Register an existing dataset.

        Args:
            name: Dataset name
            path: Path to dataset
            source: Data source type
            format: Dataset format
            classes: List of class names
            description: Dataset description

        Returns:
            DatasetInfo object
        """
        path = Path(path)

        # Count images and annotations
        num_images = 0
        num_annotations = 0
        splits = {}

        for split in ["train", "val", "test"]:
            images_dir = path / "images" / split
            labels_dir = path / "labels" / split

            if images_dir.exists():
                imgs = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
                splits[split] = len(imgs)
                num_images += len(imgs)

            if labels_dir.exists():
                labels = list(labels_dir.glob("*.txt"))
                for label_file in labels:
                    with open(label_file) as f:
                        num_annotations += len(f.readlines())

        info = DatasetInfo(
            name=name,
            version="1.0.0",
            created_date=datetime.now().isoformat(),
            source=source,
            num_images=num_images,
            num_annotations=num_annotations,
            classes=classes,
            splits=splits,
            format=format,
            path=str(path.resolve()),
            description=description,
        )

        self.datasets[name] = info
        self._save_datasets_index()

        return info

    def list_datasets(
        self,
        source: Optional[str] = None,
        format: Optional[str] = None,
    ) -> List[DatasetInfo]:
        """List registered datasets with optional filtering."""
        datasets = list(self.datasets.values())

        if source:
            datasets = [d for d in datasets if d.source == source]
        if format:
            datasets = [d for d in datasets if d.format == format]

        return datasets

    def get_dataset(self, name: str) -> Optional[DatasetInfo]:
        """Get dataset by name."""
        return self.datasets.get(name)

    def merge_datasets(
        self,
        dataset_names: List[str],
        output_name: str,
        output_format: str = "yolo",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> DatasetInfo:
        """
        Merge multiple datasets into one.

        Args:
            dataset_names: Names of datasets to merge
            output_name: Name for merged dataset
            output_format: Output format
            train_ratio: Training split ratio
            val_ratio: Validation split ratio
            test_ratio: Test split ratio

        Returns:
            DatasetInfo for merged dataset
        """
        output_dir = self.base_dir / output_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create output structure
        for split in ["train", "val", "test"]:
            (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        # Collect all files
        all_files = []
        all_classes = set()

        for name in dataset_names:
            dataset = self.datasets.get(name)
            if not dataset:
                print(f"Warning: Dataset {name} not found, skipping")
                continue

            all_classes.update(dataset.classes)

            dataset_path = Path(dataset.path)
            for split in ["train", "val", "test"]:
                images_dir = dataset_path / "images" / split
                labels_dir = dataset_path / "labels" / split

                if images_dir.exists():
                    for img_path in images_dir.iterdir():
                        if img_path.suffix.lower() in [".jpg", ".png"]:
                            label_path = labels_dir / f"{img_path.stem}.txt"
                            if label_path.exists():
                                all_files.append((img_path, label_path))

        np.random.shuffle(all_files)
        n = len(all_files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits_data = {
            "train": all_files[:n_train],
            "val": all_files[n_train : n_train + n_val],
            "test": all_files[n_train + n_val :],
        }

        for split, files in splits_data.items():
            for i, (img_path, label_path) in enumerate(tqdm(files, desc=f"Copying {split}")):
                new_name = f"merged_{i:08d}"

                shutil.copy(img_path, output_dir / "images" / split / f"{new_name}{img_path.suffix}")
                shutil.copy(label_path, output_dir / "labels" / split / f"{new_name}.txt")

        classes = sorted(list(all_classes))
        yaml_content = f"""
            path: {output_dir.resolve()}
            train: images/train
            val: images/val
            test: images/test
            names:
            """.strip()
        for i, cls in enumerate(classes):
            yaml_content += f"  {i}: {cls}\n"

        with open(output_dir / "dataset.yaml", "w") as f:
            f.write(yaml_content)

        return self.register_dataset(
            name=output_name,
            path=str(output_dir),
            source="mixed",
            format=output_format,
            classes=classes,
            description=f"Merged from: {', '.join(dataset_names)}",
        )

    def create_subset(
        self,
        dataset_name: str,
        output_name: str,
        num_samples: int,
        strategy: Literal["random", "balanced", "diverse"] = "random",
    ) -> DatasetInfo:
        """
        Create a subset of a dataset.

        Args:
            dataset_name: Source dataset name
            output_name: Output dataset name
            num_samples: Number of samples to include
            strategy: Sampling strategy

        Returns:
            DatasetInfo for subset
        """
        dataset = self.datasets.get(dataset_name)
        if not dataset:
            raise ValueError(f"Dataset {dataset_name} not found")

        output_dir = self.base_dir / output_name
        output_dir.mkdir(parents=True, exist_ok=True)

        (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)

        source_path = Path(dataset.path)

        all_images = []
        for split in ["train", "val", "test"]:
            images_dir = source_path / "images" / split
            if images_dir.exists():
                all_images.extend(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))

        if strategy == "random":
            selected = np.random.choice(all_images, min(num_samples, len(all_images)), replace=False)
        elif strategy == "balanced":
            selected = self._balanced_sample(all_images, source_path, num_samples)
        else:
            selected = self._diverse_sample(all_images, num_samples)

        for i, img_path in enumerate(tqdm(selected, desc="Creating subset")):
            label_path = None
            for split in ["train", "val", "test"]:
                candidate = source_path / "labels" / split / f"{img_path.stem}.txt"
                if candidate.exists():
                    label_path = candidate
                    break

            if label_path:
                new_name = f"subset_{i:06d}"
                shutil.copy(img_path, output_dir / "images" / "train" / f"{new_name}{img_path.suffix}")
                shutil.copy(label_path, output_dir / "labels" / "train" / f"{new_name}.txt")

        return self.register_dataset(
            name=output_name,
            path=str(output_dir),
            source=dataset.source,
            format=dataset.format,
            classes=dataset.classes,
            description=f"Subset of {dataset_name} ({strategy} sampling)",
        )

    def _balanced_sample(
        self,
        images: List[Path],
        dataset_path: Path,
        num_samples: int,
    ) -> List[Path]:
        class_to_images = {}

        for img_path in images:
            for split in ["train", "val", "test"]:
                label_path = dataset_path / "labels" / split / f"{img_path.stem}.txt"
                if label_path.exists():
                    with open(label_path) as f:
                        for line in f:
                            class_id = int(line.split()[0])
                            if class_id not in class_to_images:
                                class_to_images[class_id] = []
                            class_to_images[class_id].append(img_path)
                    break

        samples_per_class = num_samples // max(len(class_to_images), 1)
        selected = set()

        for class_id, class_images in class_to_images.items():
            class_images = list(set(class_images))
            n = min(samples_per_class, len(class_images))
            selected.update(np.random.choice(class_images, n, replace=False))

        return list(selected)[:num_samples]

    def _diverse_sample(
        self,
        images: List[Path],
        num_samples: int,
    ) -> List[Path]:
        """Sample for maximum diversity (placeholder - uses random for now)."""
        # TODO: feature-based diverse sampling
        return list(np.random.choice(images, min(num_samples, len(images)), replace=False))


class DatasetConverter:
    """
    Convert between dataset formats.

    Supports:
    - YOLO <-> COCO
    - YOLO <-> MOT
    - CVAT <-> YOLO
    """

    @staticmethod
    def yolo_to_coco(
        yolo_dir: str,
        output_path: str,
        class_names: List[str],
    ) -> str:
        """
        Convert YOLO format to COCO format.

        Args:
            yolo_dir: YOLO dataset directory
            output_path: Output COCO JSON path
            class_names: List of class names

        Returns:
            Path to COCO JSON file
        """
        import cv2

        yolo_path = Path(yolo_dir)

        coco = {
            "images": [],
            "annotations": [],
            "categories": [],
        }

        # Add categories
        for i, name in enumerate(class_names):
            coco["categories"].append(
                {
                    "id": i,
                    "name": name,
                    "supercategory": "object",
                }
            )

        annotation_id = 1
        image_id = 1

        for split in ["train", "val", "test"]:
            images_dir = yolo_path / "images" / split
            labels_dir = yolo_path / "labels" / split

            if not images_dir.exists():
                continue

            for img_path in tqdm(list(images_dir.iterdir()), desc=f"Converting {split}"):
                if img_path.suffix.lower() not in [".jpg", ".png"]:
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                h, w = img.shape[:2]

                coco["images"].append(
                    {
                        "id": image_id,
                        "file_name": str(img_path.relative_to(yolo_path)),
                        "width": w,
                        "height": h,
                    }
                )

                label_path = labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    with open(label_path) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                cx, cy, bw, bh = map(float, parts[1:5])

                                x = (cx - bw / 2) * w
                                y = (cy - bh / 2) * h
                                box_w = bw * w
                                box_h = bh * h

                                coco["annotations"].append(
                                    {
                                        "id": annotation_id,
                                        "image_id": image_id,
                                        "category_id": class_id,
                                        "bbox": [x, y, box_w, box_h],
                                        "area": box_w * box_h,
                                        "iscrowd": 0,
                                    }
                                )
                                annotation_id += 1

                image_id += 1

        with open(output_path, "w") as f:
            json.dump(coco, f, indent=2)

        return output_path

    @staticmethod
    def coco_to_yolo(
        coco_json: str,
        images_dir: str,
        output_dir: str,
    ) -> str:
        """
        Convert COCO format to YOLO format.

        Args:
            coco_json: Path to COCO annotations JSON
            images_dir: Directory containing images
            output_dir: Output YOLO directory

        Returns:
            Path to output directory
        """
        with open(coco_json) as f:
            coco = json.load(f)

        output_path = Path(output_dir)
        labels_dir = output_path / "labels" / "train"
        labels_dir.mkdir(parents=True, exist_ok=True)

        image_info = {img["id"]: img for img in coco["images"]}

        annotations_by_image = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)

        for img_id, annotations in tqdm(annotations_by_image.items(), desc="Converting"):
            img_info = image_info.get(img_id)
            if not img_info:
                continue

            w, h = img_info["width"], img_info["height"]
            img_name = Path(img_info["file_name"]).stem

            lines = []
            for ann in annotations:
                class_id = ann["category_id"]
                x, y, bw, bh = ann["bbox"]

                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                nw = bw / w
                nh = bh / h

                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            with open(labels_dir / f"{img_name}.txt", "w") as f:
                f.write("\n".join(lines))

        images_out = output_path / "images" / "train"
        images_out.mkdir(parents=True, exist_ok=True)

        for img_info in coco["images"]:
            src = Path(images_dir) / img_info["file_name"]
            if src.exists():
                shutil.copy(src, images_out / src.name)

        class_names = {cat["id"]: cat["name"] for cat in coco["categories"]}
        yaml_content = f"""
            path: {output_path.resolve()}
            train: images/train
            val: images/train
            names:
            """.strip()

        for i in sorted(class_names.keys()):
            yaml_content += f"  {i}: {class_names[i]}\n"

        with open(output_path / "dataset.yaml", "w") as f:
            f.write(yaml_content)

        return str(output_path)

    @staticmethod
    def mot_to_yolo(
        mot_dir: str,
        output_dir: str,
        class_mapping: Optional[Dict[int, int]] = None,
    ) -> str:
        """
        Convert MOT format to YOLO format.

        Args:
            mot_dir: MOT dataset directory (with gt/gt.txt and img1/)
            output_dir: Output YOLO directory
            class_mapping: Optional mapping from MOT class to YOLO class

        Returns:
            Path to output directory
        """
        import cv2
        import pandas as pd

        mot_path = Path(mot_dir)
        output_path = Path(output_dir)

        images_dir = output_path / "images" / "train"
        labels_dir = output_path / "labels" / "train"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        gt_path = mot_path / "gt" / "gt.txt"
        if not gt_path.exists():
            raise ValueError(f"Ground truth not found: {gt_path}")

        df = pd.read_csv(
            gt_path,
            header=None,
            names=["frame", "id", "x", "y", "w", "h", "conf", "class", "vis", "unused"],
        )

        img1_dir = mot_path / "img1"
        first_img = list(img1_dir.glob("*.jpg"))[0]
        sample = cv2.imread(str(first_img))
        img_h, img_w = sample.shape[:2]

        for frame_num in tqdm(df["frame"].unique(), desc="Converting MOT"):
            frame_df = df[df["frame"] == frame_num]

            img_name = f"{frame_num:06d}.jpg"
            src_img = img1_dir / img_name
            if src_img.exists():
                shutil.copy(src_img, images_dir / img_name)

            # Write labels
            lines = []
            for _, row in frame_df.iterrows():
                mot_class = int(row["class"])
                class_id = class_mapping.get(mot_class, 0) if class_mapping else 0

                # Convert to YOLO format
                x, y, w, h = row["x"], row["y"], row["w"], row["h"]
                cx = (x + w / 2) / img_w
                cy = (y + h / 2) / img_h
                nw = w / img_w
                nh = h / img_h

                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            label_name = f"{frame_num:06d}.txt"
            with open(labels_dir / label_name, "w") as f:
                f.write("\n".join(lines))

        return str(output_path)


class ActiveLearningPipeline:
    """
    Active learning pipeline for iterative model improvement.

    Workflow:
    1. Run model predictions on new data
    2. Identify uncertain/difficult samples
    3. Upload to CVAT for annotation
    4. Export annotations and retrain
    """

    def __init__(
        self,
        dataset_manager: DatasetManager,
        model_path: str,
        cvat_client=None,
    ):
        """
        Initialize active learning pipeline.

        Args:
            dataset_manager: Dataset manager instance
            model_path: Path to current model
            cvat_client: Optional CVAT client for annotation upload
        """
        self.dataset_manager = dataset_manager
        self.model_path = model_path
        self.cvat_client = cvat_client

        self.model = None

    def _load_model(self):
        """Load the detection model."""
        if self.model is None:
            from ultralytics import YOLO

            self.model = YOLO(self.model_path)

    def identify_uncertain_samples(
        self,
        images_dir: str,
        top_k: int = 100,
        uncertainty_threshold: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """
        Identify samples where model is uncertain.

        Args:
            images_dir: Directory of images to analyze
            top_k: Number of most uncertain samples to return
            uncertainty_threshold: Confidence threshold for uncertainty

        Returns:
            List of (image_path, uncertainty_score) tuples
        """
        self._load_model()

        images_path = Path(images_dir)
        images = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))

        uncertain_samples = []

        for img_path in tqdm(images, desc="Analyzing uncertainty"):
            results = self.model(str(img_path), verbose=False)[0]

            # Compute uncertainty score
            if len(results.boxes) == 0:
                # No detections - might be hard sample
                uncertainty = 0.8
            else:
                # Use average confidence as inverse uncertainty
                confidences = results.boxes.conf.cpu().numpy()

                # Samples with many low-confidence detections are uncertain
                low_conf_ratio = np.sum(confidences < uncertainty_threshold) / len(confidences)
                avg_conf = np.mean(confidences)

                uncertainty = 0.5 * low_conf_ratio + 0.5 * (1 - avg_conf)

            uncertain_samples.append((str(img_path), uncertainty))

        # Sort by uncertainty and return top-k
        uncertain_samples.sort(key=lambda x: x[1], reverse=True)

        return uncertain_samples[:top_k]

    def identify_hard_negatives(
        self,
        images_dir: str,
        labels_dir: str,
        iou_threshold: float = 0.5,
        top_k: int = 100,
    ) -> List[Tuple[str, Dict]]:
        """
        Identify samples with high false positive/negative rates.

        Args:
            images_dir: Directory of images
            labels_dir: Directory of ground truth labels
            iou_threshold: IoU threshold for matching
            top_k: Number of hardest samples

        Returns:
            List of (image_path, error_info) tuples
        """
        self._load_model()

        images_path = Path(images_dir)
        labels_path = Path(labels_dir)

        hard_samples = []

        for img_path in tqdm(list(images_path.iterdir()), desc="Finding hard samples"):
            if img_path.suffix.lower() not in [".jpg", ".png"]:
                continue

            label_path = labels_path / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue

            results = self.model(str(img_path), verbose=False)[0]
            pred_boxes = results.boxes.xyxy.cpu().numpy() if len(results.boxes) > 0 else np.array([])

            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]

            gt_boxes = []
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cx, cy, bw, bh = map(float, parts[1:5])
                        x1 = (cx - bw / 2) * w
                        y1 = (cy - bh / 2) * h
                        x2 = (cx + bw / 2) * w
                        y2 = (cy + bh / 2) * h
                        gt_boxes.append([x1, y1, x2, y2])

            gt_boxes = np.array(gt_boxes)

            n_gt = len(gt_boxes)
            n_pred = len(pred_boxes)

            if n_gt == 0 and n_pred == 0:
                continue

            false_positives = max(0, n_pred - n_gt)
            false_negatives = max(0, n_gt - n_pred)

            error_score = (false_positives + false_negatives) / max(n_gt, n_pred, 1)

            if error_score > 0:
                hard_samples.append(
                    (
                        str(img_path),
                        {
                            "false_positives": false_positives,
                            "false_negatives": false_negatives,
                            "error_score": error_score,
                            "n_gt": n_gt,
                            "n_pred": n_pred,
                        },
                    )
                )

        hard_samples.sort(key=lambda x: x[1]["error_score"], reverse=True)

        return hard_samples[:top_k]

    def create_annotation_task(
        self,
        image_paths: List[str],
        task_name: str,
        project_id: Optional[int] = None,
    ) -> int:
        """
        Create CVAT annotation task for selected samples.

        Args:
            image_paths: Paths to images for annotation
            task_name: Name for the CVAT task
            project_id: Optional CVAT project ID

        Returns:
            CVAT task ID
        """
        if self.cvat_client is None:
            raise ValueError("CVAT client not configured")

        task = self.cvat_client.create_task(
            name=task_name,
            project_id=project_id,
        )

        self.cvat_client.upload_images_to_task(task.id, image_paths)

        if self.model is not None:
            from cvat.cvat_integration import ModelAssistedLabeling

            mal = ModelAssistedLabeling(detection_model_path=self.model_path)
            mal.upload_predictions_to_cvat(
                self.cvat_client,
                task.id,
                image_paths,
                confidence_threshold=0.3,
            )

        return task.id
