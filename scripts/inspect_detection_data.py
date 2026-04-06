"""
Quick inspection of detection training dataloader.
Saves 16 sample images with ground-truth boxes drawn to /workspace/inspect_samples/.

Usage:
    python scripts/inspect_detection_data.py \
        --soccernet-dir /tmp/soccernet_extracted/ \
        --output-dir /workspace/inspect_samples/
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader


def _collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets


def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.permute(1, 2, 0).numpy()
    img = img * std + mean
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--soccernet-dir", required=True)
    parser.add_argument("--output-dir", default="/workspace/inspect_samples/")
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--augment", action="store_true")
    args = parser.parse_args()

    from torchkick.training.data.detection_dataset import MixedDetectionDataset

    data_config = [{"type": "soccernet_dir", "path": args.soccernet_dir}]
    ds = MixedDetectionDataset(data_config, input_size=640, augment=args.augment)
    loader = DataLoader(ds, batch_size=args.n_samples, shuffle=True, collate_fn=_collate_fn)

    images, targets = next(iter(loader))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, (img_t, tgt) in enumerate(zip(images, targets)):
        img = denormalize(img_t)
        boxes = tgt["boxes"].numpy()  # xyxy, pixel coords
        labels = tgt["labels"].numpy()

        print(
            f"  sample {i:02d}: {len(boxes)} boxes  "
            f"box sizes: {[(int(b[2]-b[0]), int(b[3]-b[1])) for b in boxes[:5]]}"
        )

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Stats overlay
        cv2.putText(img, f"n={len(boxes)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imwrite(str(out_dir / f"sample_{i:02d}.jpg"), img)

    print(f"\nSaved {args.n_samples} samples to {args.output_dir}")
    print(f"Dataset size: {len(ds)}")

    # Summary stats
    all_boxes = [tgt["boxes"] for tgt in targets]
    box_counts = [len(b) for b in all_boxes]
    all_wh = [(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]) for b in all_boxes if len(b)]
    print(f"Boxes per image: min={min(box_counts)} max={max(box_counts)} mean={np.mean(box_counts):.1f}")
    if all_wh:
        ws = torch.cat([w for w, h in all_wh])
        hs = torch.cat([h for w, h in all_wh])
        print(f"Box width:  min={ws.min():.1f} max={ws.max():.1f} mean={ws.mean():.1f}")
        print(f"Box height: min={hs.min():.1f} max={hs.max():.1f} mean={hs.mean():.1f}")


if __name__ == "__main__":
    main()
