import cv2
import fsspec
import torch
import json
import numpy as np
from io import BytesIO
from PIL import Image
from typing import Literal
from torch.utils.data import Dataset

LINE_CLASSES = [
    "Big rect. left bottom",
    "Big rect. left main",
    "Big rect. left top",
    "Big rect. right bottom",
    "Big rect. right main",
    "Big rect. right top",
    "Circle central",
    "Circle left",
    "Circle right",
    "Goal left crossbar",
    "Goal left post left ",
    "Goal left post right",
    "Goal right crossbar",
    "Goal right post left",
    "Goal right post right",
    "Goal unknown",
    "Line unknown",
    "Middle line",
    "Side line bottom",
    "Side line left",
    "Side line right",
    "Side line top",
    "Small rect. left bottom",
    "Small rect. left main",
    "Small rect. left top",
    "Small rect. right bottom",
    "Small rect. right main",
    "Small rect. right top",
]

CIRCLE_CLASSES = {"Circle central", "Circle left", "Circle right"}


class LineDetectionDataset(Dataset):
    """
    Dataset for soccer pitch line detection from SoccerNet calibration data.

    Annotations are keypoints (x, y in normalized [0,1] coords) for each line class.
    Lines have 2 endpoints, circles have ~9 points.

    Output modes:
    - "segmentation": Returns [H, W] mask with class indices (for DeepLabV3)
    - "keypoints": Returns dict of class -> [(x, y), ...] normalized coords
    - "heatmap": Returns [C, H, W] gaussian heatmaps per class (for HRNet-style)
    """

    def __init__(
        self,
        zip_path: str,
        output_mode: Literal["segmentation", "keypoints", "heatmap"] = "keypoints",
        transform=None,
        width: int = 960,
        height: int = 540,
        line_thickness: int = 3,
        heatmap_sigma: float = 2.0,
    ):
        self.zip_path = zip_path
        self.output_mode = output_mode
        self.transform = transform
        self.width = width
        self.height = height
        self.line_thickness = line_thickness
        self.heatmap_sigma = heatmap_sigma

        self.samples = []
        self._build_index()

    def _build_index(self) -> None:
        fs, _, _ = fsspec.get_fs_token_paths(self.zip_path)

        with fs.open(self.zip_path, "rb") as f:
            zip_fs = fsspec.filesystem("zip", fo=f)

            all_files = zip_fs.ls("", detail=False)
            root_dirs = [p for p in all_files if zip_fs.isdir(p)]
            root = root_dirs[0].rstrip("/") if root_dirs else ""

            files = zip_fs.ls(f"{root}/" if root else "", detail=False)
            json_files = sorted([f for f in files if f.endswith(".json")])

            for json_path in json_files:
                frame_id = json_path.split("/")[-1].replace(".json", "")
                img_path = json_path.replace(".json", ".jpg")

                if zip_fs.exists(img_path):
                    with zip_fs.open(json_path, "r") as jf:
                        annotations = json.load(jf)

                    if annotations:
                        self.samples.append(
                            {
                                "frame_id": frame_id,
                                "img_path": img_path,
                                "annotations": annotations,
                            }
                        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, img_path: str) -> np.ndarray:
        full_path = f"zip://{img_path}::{self.zip_path}"
        with fsspec.open(full_path, "rb") as f:
            img = Image.open(BytesIO(f.read())).convert("RGB")
        img = img.resize((self.width, self.height), Image.BILINEAR)
        return np.array(img)

    def _create_segmentation_mask(self, annotations: dict, orig_h: int, orig_w: int) -> np.ndarray:
        mask = np.zeros((self.height, self.width), dtype=np.uint8)

        for class_idx, class_name in enumerate(LINE_CLASSES):
            if class_name not in annotations:
                continue

            points = annotations[class_name]
            if len(points) < 2:
                continue

            pts = []
            for p in points:
                x = int(p["x"] * self.width)
                y = int(p["y"] * self.height)
                pts.append((x, y))

            for i in range(len(pts) - 1):
                cv2.line(
                    mask,
                    pts[i],
                    pts[i + 1],
                    class_idx + 1,
                    self.line_thickness,
                )

        return mask

    def _create_heatmaps(self, annotations: dict) -> np.ndarray:
        num_classes = len(LINE_CLASSES)
        heatmaps = np.zeros((num_classes, self.height, self.width), dtype=np.float32)

        sigma = self.heatmap_sigma
        size = int(6 * sigma + 1)
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0, y0 = size // 2, size // 2
        gaussian = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))

        for class_idx, class_name in enumerate(LINE_CLASSES):
            if class_name not in annotations:
                continue

            points = annotations[class_name]
            for p in points:
                px = int(p["x"] * self.width)
                py = int(p["y"] * self.height)

                ul = [int(px - size // 2), int(py - size // 2)]
                br = [int(px + size // 2 + 1), int(py + size // 2 + 1)]

                c, d = max(0, -ul[0]), min(br[0], self.width) - ul[0]
                a, b = max(0, -ul[1]), min(br[1], self.height) - ul[1]

                cc, dd = max(0, ul[0]), min(br[0], self.width)
                aa, bb = max(0, ul[1]), min(br[1], self.height)

                if aa < bb and cc < dd and a < b and c < d:
                    heatmaps[class_idx, aa:bb, cc:dd] = np.maximum(
                        heatmaps[class_idx, aa:bb, cc:dd], gaussian[a:b, c:d]
                    )

        return heatmaps

    def _extract_keypoints(self, annotations: dict) -> dict:
        keypoints = {}
        for class_name in LINE_CLASSES:
            if class_name in annotations:
                points = annotations[class_name]
                keypoints[class_name] = [(p["x"], p["y"]) for p in points]
            else:
                keypoints[class_name] = []
        return keypoints

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        img = self._load_image(sample["img_path"])
        annotations = sample["annotations"]

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        if self.transform:
            img_tensor = self.transform(img_tensor)

        result = {
            "image": img_tensor,
            "frame_id": sample["frame_id"],
        }

        if self.output_mode == "segmentation":
            mask = self._create_segmentation_mask(annotations, img.shape[0], img.shape[1])
            result["mask"] = torch.from_numpy(mask).long()

        elif self.output_mode == "heatmap":
            heatmaps = self._create_heatmaps(annotations)
            result["heatmaps"] = torch.from_numpy(heatmaps)

        elif self.output_mode == "keypoints":
            result["keypoints"] = self._extract_keypoints(annotations)
            result["num_classes"] = len(LINE_CLASSES)

        return result


class LineKeypointDataset(Dataset):
    """
    Simplified dataset that outputs fixed-size tensors for keypoint regression.

    For each class:
    - Non-circle classes: 2 keypoints (endpoints)
    - Circle classes: up to MAX_CIRCLE_POINTS keypoints

    Output:
    - keypoints: [num_classes, max_points, 2] - normalized (x, y)
    - visibility: [num_classes, max_points] - 1 if point exists, 0 otherwise
    """

    MAX_LINE_POINTS = 2
    MAX_CIRCLE_POINTS = 12

    def __init__(
        self,
        zip_path: str,
        transform=None,
        width: int = 960,
        height: int = 540,
    ):
        self.zip_path = zip_path
        self.transform = transform
        self.width = width
        self.height = height
        self.num_classes = len(LINE_CLASSES)

        self.samples = []
        self._build_index()

    def _build_index(self) -> None:
        fs, _, _ = fsspec.get_fs_token_paths(self.zip_path)

        with fs.open(self.zip_path, "rb") as f:
            zip_fs = fsspec.filesystem("zip", fo=f)

            all_files = zip_fs.ls("", detail=False)
            root_dirs = [p for p in all_files if zip_fs.isdir(p)]
            root = root_dirs[0].rstrip("/") if root_dirs else ""

            files = zip_fs.ls(f"{root}/" if root else "", detail=False)
            json_files = sorted([f for f in files if f.endswith(".json")])

            for json_path in json_files:
                frame_id = json_path.split("/")[-1].replace(".json", "")
                img_path = json_path.replace(".json", ".jpg")

                if zip_fs.exists(img_path):
                    with zip_fs.open(json_path, "r") as jf:
                        annotations = json.load(jf)

                    if annotations:
                        self.samples.append(
                            {
                                "frame_id": frame_id,
                                "img_path": img_path,
                                "annotations": annotations,
                            }
                        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, img_path: str) -> np.ndarray:
        full_path = f"zip://{img_path}::{self.zip_path}"
        with fsspec.open(full_path, "rb") as f:
            img = Image.open(BytesIO(f.read())).convert("RGB")
        img = img.resize((self.width, self.height), Image.BILINEAR)
        return np.array(img)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        img = self._load_image(sample["img_path"])
        annotations = sample["annotations"]

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        if self.transform:
            img_tensor = self.transform(img_tensor)

        max_pts = self.MAX_CIRCLE_POINTS
        keypoints = torch.zeros((self.num_classes, max_pts, 2), dtype=torch.float32)
        visibility = torch.zeros((self.num_classes, max_pts), dtype=torch.float32)

        for class_idx, class_name in enumerate(LINE_CLASSES):
            if class_name not in annotations:
                continue

            points = annotations[class_name]
            is_circle = class_name in CIRCLE_CLASSES

            # Sort points for lines to ensure canonical order (left-to-right)
            # This prevents the model from being confused by random point ordering in GT
            if not is_circle and len(points) > 1:
                points = sorted(points, key=lambda p: (p["x"], p["y"]))

            max_allowed = max_pts if is_circle else self.MAX_LINE_POINTS
            for i, p in enumerate(points[:max_allowed]):
                keypoints[class_idx, i, 0] = p["x"]
                keypoints[class_idx, i, 1] = p["y"]
                visibility[class_idx, i] = 1.0

        return {
            "image": img_tensor,
            "keypoints": keypoints,
            "visibility": visibility,
            "frame_id": sample["frame_id"],
        }


def visualize_sample(sample: dict, show_labels: bool = True) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    img = sample["image"].permute(1, 2, 0).numpy()
    h, w = img.shape[:2]

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.imshow(img)

    if "mask" in sample:
        mask = sample["mask"].numpy()
        ax.imshow(mask, alpha=0.4, cmap="jet")

    if "keypoints" in sample:
        keypoints: torch.Tensor = sample["keypoints"]
        visibility = sample.get("visibility", None)

        cmap = plt.cm.get_cmap("tab20", len(LINE_CLASSES))
        legend_handles = []

        for class_idx, class_name in enumerate(LINE_CLASSES):
            points = keypoints[class_idx]
            color = cmap(class_idx)

            if visibility is not None:
                vis = visibility[class_idx]
                valid_pts = [(p[0].item() * w, p[1].item() * h) for i, p in enumerate(points) if vis[i] > 0.5]
            else:
                valid_pts = [(p[0].item() * w, p[1].item() * h) for p in points if p[0] > 0 or p[1] > 0]

            if not valid_pts:
                continue

            xs = [p[0] for p in valid_pts]
            ys = [p[1] for p in valid_pts]

            is_circle = class_name in CIRCLE_CLASSES
            if is_circle and len(valid_pts) >= 3:
                xs_closed = xs + [xs[0]]
                ys_closed = ys + [ys[0]]
                ax.plot(xs_closed, ys_closed, "-", color=color, linewidth=2, alpha=0.8)
            elif len(valid_pts) >= 2:
                ax.plot(xs, ys, "-", color=color, linewidth=2, alpha=0.8)

            ax.scatter(xs, ys, c=[color], s=40, edgecolors="white", linewidths=0.5, zorder=5)

            if show_labels:
                legend_handles.append(mpatches.Patch(color=color, label=class_name))

        if show_labels and legend_handles:
            ax.legend(handles=legend_handles, loc="upper left", fontsize=6, bbox_to_anchor=(1.01, 1), borderaxespad=0)

    ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    zip_path = "soccernet/calibration/calibration/train.zip"
    ds_kp = LineKeypointDataset(zip_path)
    ds_seg = LineDetectionDataset(zip_path, output_mode="segmentation")

    # visualize a line keypoint sample
    print("\nVisualizing a sample from LineKeypointDataset...")
    visualize_sample(ds_kp[0])
    # visualize a mask sample from segmentation dataset
    print("\nVisualizing a sample from LineDetectionDataset (segmentation)...")
    visualize_sample(ds_seg[0])
