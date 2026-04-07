"""
Inspect detection dataset: scan all gt.txt files and report raw class IDs,
then show sample images with ground-truth boxes drawn.

Usage:
    python scripts/inspect_detection_data.py \
        --soccernet-dir /tmp/soccernet_extracted/ \
        --output-dir /workspace/inspect_samples/
"""

import argparse
from collections import defaultdict
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
    return (img * 255).clip(0, 255).astype(np.uint8)


def scan_class_ids(soccernet_dir: str) -> None:
    """Scan all gt.txt files and report every raw class_id seen."""
    root = Path(soccernet_dir)
    global_counts: dict = defaultdict(int)
    seq_summary = {}

    for gt_file in sorted(root.glob("**/gt/gt.txt")):
        seq_name = gt_file.parts[-3]
        local_counts: dict = defaultdict(int)
        with open(gt_file) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 8:
                    continue
                if parts[6].strip() == "0":  # conf=0 → ignore
                    continue
                class_id = parts[7].strip()
                local_counts[class_id] += 1
                global_counts[class_id] += 1
        seq_summary[seq_name] = dict(local_counts)

    print("\n=== RAW CLASS ID SCAN ===")
    print(f"Sequences scanned: {len(seq_summary)}")
    print(f"\nGlobal class_id counts (across all sequences):")
    for cid, count in sorted(global_counts.items(), key=lambda x: -x[1]):
        print(f"  class_id={cid!r:6s}  {count:>8d} annotations")

    print(f"\nPer-sequence breakdown:")
    for seq, counts in seq_summary.items():
        print(f"  {seq}: {dict(sorted(counts.items()))}")
    print("=========================\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--soccernet-dir", required=True)
    parser.add_argument("--output-dir", default="/workspace/inspect_samples/")
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--scan-only", action="store_true", help="Only scan class IDs, skip image output.")
    args = parser.parse_args()

    # Always scan class IDs first
    scan_class_ids(args.soccernet_dir)

    if args.scan_only:
        return

    from torchkick.training.data.detection_dataset import MixedDetectionDataset

    data_config = [{"type": "soccernet_dir", "path": args.soccernet_dir}]
    ds = MixedDetectionDataset(data_config, input_size=640, augment=args.augment)
    print(f"Dataset size after loading: {len(ds)}")

    loader = DataLoader(ds, batch_size=args.n_samples, shuffle=True, collate_fn=_collate_fn)
    images, targets = next(iter(loader))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_colors = {0: (0, 255, 0), 1: (0, 128, 255)}  # green=player, orange=ball
    label_names = {0: "player", 1: "ball"}

    for i, (img_t, tgt) in enumerate(zip(images, targets)):
        img = cv2.cvtColor(denormalize(img_t), cv2.COLOR_RGB2BGR)
        boxes = tgt["boxes"].numpy()
        labels = tgt["labels"].numpy()

        label_dist = {int(l): int((labels == l).sum()) for l in np.unique(labels)}
        print(
            f"  sample {i:02d}: {len(boxes)} boxes  labels={label_dist}  "
            f"sizes={[(int(b[2]-b[0]), int(b[3]-b[1])) for b in boxes[:4]]}"
        )

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = map(int, box)
            color = label_colors.get(int(label), (255, 0, 0))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img,
                label_names.get(int(label), str(label)),
                (x1, max(y1 - 4, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

        cv2.putText(img, f"n={len(boxes)} {label_dist}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(str(out_dir / f"sample_{i:02d}.jpg"), img)

    print(f"\nSaved {min(args.n_samples, len(ds))} samples to {args.output_dir}")

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
