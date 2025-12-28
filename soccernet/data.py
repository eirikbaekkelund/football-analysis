import cv2
import fsspec
import torch
import pandas as pd
import numpy as np
from io import BytesIO
from PIL import Image
from typing import List, Dict, Any, Literal
from torch.utils.data import Dataset

class SoccerNetTrackingDataset(Dataset):
    def __init__(
        self, 
        zip_path: str, 
        transform=None,
        bbox_format: Literal["xywh", "xyxy", "cxcywh"] = "xyxy",
        debug: bool = False
    ):
        """
        Args:
            zip_path: Path to the zip file
            transform: Optional transform to apply to images
            bbox_format: Output format for bounding boxes
                - "xywh": [x, y, w, h] - top-left corner + dimensions (MOT format)
                - "xyxy": [x1, y1, x2, y2] - top-left + bottom-right corners
                - "cxcywh": [cx, cy, w, h] - center + dimensions
            debug: Print debug information about loaded annotations
        """
        self.zip_path = zip_path
        self.transform = transform
        self.bbox_format = bbox_format
        self.debug = debug

        self.samples = []
        self.annotations = {}

        self._build_index()

    def _build_index(self):
        fs, _, _ = fsspec.get_fs_token_paths(self.zip_path)

        with fs.open(self.zip_path, "rb") as f:
            zip_fs = fsspec.filesystem("zip", fo=f)

            # root folder (train/, test/, val/)
            root_dirs = [
                p["name"] if isinstance(p, dict) else p
                for p in zip_fs.ls("")
            ]
            # we expect exactly one root dir s.t. we index by 0-th elmnt
            root = root_dirs[0].rstrip("/")

            sequences_dir = f"{root}/"

            for seq in zip_fs.ls(sequences_dir):
                seq_name = seq["name"] if isinstance(seq, dict) else seq
                seq_name = seq_name.rstrip("/")

                gt_path = f"{seq_name}/gt/gt.txt"
                if not zip_fs.exists(gt_path):
                    continue

                with zip_fs.open(gt_path) as gt:
                    # NOTE: format: frame,track_id,x,y,w,h,conf,class_id,visibility,unused
                    df = pd.read_csv(
                        gt,
                        header=None,
                        names=[
                            "frame",
                            "track_id",
                            "x",
                            "y",
                            "w",
                            "h",
                            "conf",
                            "class_id",
                            "visibility",
                            "unused",
                        ],
                    )
                    
                self.annotations[seq_name] = df

                for frame in df["frame"].unique():
                    self.samples.append((seq_name, frame))
        
    def __len__(self):
        return len(self.samples)

    def _load_image(self, seq, frame):
        img_path = f"zip://{seq}/img1/{frame:06d}.jpg::{self.zip_path}"
        with fsspec.open(img_path, "rb") as f:
            img: Image.Image = Image.open(BytesIO(f.read())).convert("RGB")
        
        # Convert to tensor
        img = torch.from_numpy(
            np.array(img)
        ).permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        return img

    def _convert_bbox_format(self, boxes: np.ndarray) -> np.ndarray:
        """
        Convert bounding boxes from MOT format (xywh) to desired format.
        
        Args:
            boxes: Array of shape [N, 4] in format [x, y, w, h]
        
        Returns:
            Converted boxes in the specified format
        """
        if len(boxes) == 0:
            return boxes
        
        boxes = boxes.astype(np.float32).copy()
        
        if self.bbox_format == "xywh":
            return boxes
        
        elif self.bbox_format == "xyxy":
            converted = np.zeros_like(boxes)
            converted[:, 0] = boxes[:, 0]
            converted[:, 1] = boxes[:, 1]
            converted[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2 = x + w
            converted[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2 = y + h
            return converted
        
        elif self.bbox_format == "cxcywh":
            converted = np.zeros_like(boxes)
            converted[:, 0] = boxes[:, 0] + boxes[:, 2] / 2  # cx = x + w/2
            converted[:, 1] = boxes[:, 1] + boxes[:, 3] / 2  # cy = y + h/2
            converted[:, 2] = boxes[:, 2]                    # w stays the same
            converted[:, 3] = boxes[:, 3]                    # h stays the same
            return converted
        
        else:
            raise ValueError(f"Unknown bbox format: {self.bbox_format}")

    def _jersey_color_feature(self, image: torch.Tensor, box: np.ndarray):
        """
        Extract jersey color features from a bounding box.
        
        Args:
            image: Tensor [3, H, W] (after transform)
            box: [x, y, w, h] in pixel coords (original MOT format)
        
        Returns:
            3D HSV mean vector
        """
        x, y, w, h = box.astype(int)

        _, H, W = image.size()
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(W, x + w)
        y2 = min(H, y + h)

        if x2 <= x1 or y2 <= y1:
            return torch.zeros(3)

        crop = image[:, y1:y2, x1:x2]
        if crop.numel() == 0:
            return torch.zeros(3)

        crop_np = crop.permute(1, 2, 0).cpu().numpy()

        h_crop = crop_np.shape[0]
        crop_np = crop_np[: int(0.6 * h_crop)]

        if crop_np.size == 0:
            return torch.zeros(3)

        crop_np = (crop_np * 255).astype(np.uint8) if crop_np.max() <= 1.0 else crop_np.astype(np.uint8)

        hsv = cv2.cvtColor(crop_np, cv2.COLOR_RGB2HSV)
        mean_hsv = hsv.reshape(-1, 3).mean(axis=0)

        return torch.tensor(mean_hsv, dtype=torch.float32)
    
    def __getitem__(self, idx):
        seq, frame = self.samples[idx]
        image = self._load_image(seq, frame)

        ann = self.annotations[seq]
        ann = ann[ann.frame == frame].copy()

        boxes_mot = ann[["x", "y", "w", "h"]].values.astype(np.float32)
        track_ids = ann["track_id"].values.astype(np.int64)

        if self.transform:
            image = self.transform(image)
        
        color_features = [
            self._jersey_color_feature(image, box) 
            for box in boxes_mot
        ]

        boxes = self._convert_bbox_format(boxes_mot)
        
        return {
            "image": image,
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "track_ids": torch.tensor(track_ids, dtype=torch.long),
            "color_features": torch.stack(color_features) if color_features else torch.zeros((0, 3)),
            "seq_name": seq,
            "frame": frame,
        }

def tracking_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for batching tracking data.
    """
    return {
        "images": torch.stack([b["image"] for b in batch]),
        "boxes": [b["boxes"] for b in batch],
        "track_ids": [b["track_ids"] for b in batch],
        "color_features": [b["color_features"] for b in batch],
        "seq_names": [b["seq_name"] for b in batch],
        "frames": [b["frame"] for b in batch],
    }

def visualize_sample(sample: Dict[str, Any], bbox_format: str = "xyxy", show_diagnostics: bool = True) -> None:
    """
    Visualize a sample with bounding boxes.
    
    Args:
        sample: Dictionary containing image, boxes, track_ids
        bbox_format: Format of the bounding boxes in the sample
        show_diagnostics: Print diagnostic information about boxes
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    image = sample["image"].permute(1, 2, 0).cpu().numpy()
    boxes = sample["boxes"].cpu().numpy()
    track_ids = sample["track_ids"].cpu().numpy()

    img_h, img_w = image.shape[:2]


    fig, ax = plt.subplots(1, figsize=(16, 10))
    ax.imshow(image)

    for box, track_id in zip(boxes, track_ids):
        if bbox_format == "xywh":
            x, y, w, h = box
        elif bbox_format == "xyxy":
            x, y, x2, y2 = box
            w, h = x2 - x, y2 - y
        elif bbox_format == "cxcywh":
            cx, cy, w, h = box
            x, y = cx - w/2, cy - h/2
        else:
            raise ValueError(f"Unknown bbox format: {bbox_format}")
        
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(x, y - 10, f'ID: {track_id}', 
                color='yellow', fontsize=10, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))

    ax.set_title(f"Frame {sample.get('frame', 'N/A')} - Seq: {sample.get('seq_name', 'N/A')}\nImage: {img_w}x{img_h}")
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)  
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    zip_path = "soccernet/tracking/tracking/train.zip"
    
    dataset_xywh = SoccerNetTrackingDataset(zip_path, bbox_format="xywh", debug=True)
    sample_xywh = dataset_xywh[0]
    visualize_sample(sample_xywh, bbox_format="xywh", show_diagnostics=True)
    
    dataset_xyxy = SoccerNetTrackingDataset(zip_path, bbox_format="xyxy", debug=False)
    sample_xyxy = dataset_xyxy[0]
    visualize_sample(sample_xyxy, bbox_format="xyxy", show_diagnostics=True)