"""
Post-hoc evaluation of a saved RT-DETR detection checkpoint.

Sweeps confidence thresholds and reports MAP@0.5, score distribution,
and predictions-per-image at each threshold. Helps diagnose whether
MAP collapse is a threshold artifact or real regression.

Usage:
    python scripts/eval_detection.py \
        --checkpoint weights/detection_soccernet_only/rtdetr_best.pth \
        --soccernet-dir /tmp/soccernet_extracted/ \
        --num-classes 1
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split


def _collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--soccernet-dir", required=True)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--model-name", default="PekingU/rtdetr_r101vd")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    input_size = 640

    # Dataset
    from torchkick.training.data.detection_dataset import MixedDetectionDataset

    data_config = [{"type": "soccernet_dir", "path": args.soccernet_dir}]
    full_ds = MixedDetectionDataset(data_config, input_size=input_size, augment=False)
    n_val = max(1, int(len(full_ds) * args.val_split))
    n_train = len(full_ds) - n_val
    generator = torch.Generator().manual_seed(42)
    val_ds, _ = random_split(full_ds, [n_val, n_train], generator=generator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, collate_fn=_collate_fn)
    print(f"Val set: {n_val} samples")

    # Model
    from transformers import RTDetrForObjectDetection
    model = RTDetrForObjectDetection.from_pretrained(
        args.model_name, num_labels=args.num_classes, ignore_mismatched_sizes=True
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model = model.to(dev).eval()
    print(f"Loaded checkpoint: {args.checkpoint}  (epoch={ckpt.get('epoch', '?')}  map50={ckpt.get('map50', '?'):.4f})")

    # Collect all scores and boxes
    all_scores = []   # list of [300] tensors
    all_pred_boxes = []  # list of [300, 4] absolute xyxy
    all_gt_boxes = []
    all_gt_labels = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(dev)
            hf_labels = []
            for t in targets:
                boxes = t["boxes"].to(dev)
                labels_t = t["labels"].to(dev)
                if len(boxes):
                    cx = (boxes[:, 0] + boxes[:, 2]) / 2 / input_size
                    cy = (boxes[:, 1] + boxes[:, 3]) / 2 / input_size
                    bw = (boxes[:, 2] - boxes[:, 0]) / input_size
                    bh = (boxes[:, 3] - boxes[:, 1]) / input_size
                    valid = (bw > 0) & (bh > 0)
                    boxes_n = torch.stack([cx, cy, bw, bh], dim=1)[valid]
                    labels_t = labels_t[valid]
                else:
                    boxes_n = boxes
                hf_labels.append({"class_labels": labels_t, "boxes": boxes_n})

            out = model(pixel_values=images, labels=hf_labels)
            scores = out.logits.sigmoid()[:, :, 0].cpu()  # [B, 300]
            pred_boxes = out.pred_boxes.cpu()             # [B, 300, 4] norm cxcywh

            for i in range(len(targets)):
                s = scores[i]
                pb = pred_boxes[i]
                cx, cy, bw, bh = pb.unbind(-1)
                abs_boxes = torch.stack([
                    (cx - bw / 2) * input_size,
                    (cy - bh / 2) * input_size,
                    (cx + bw / 2) * input_size,
                    (cy + bh / 2) * input_size,
                ], dim=-1)
                all_scores.append(s)
                all_pred_boxes.append(abs_boxes)
                all_gt_boxes.append(targets[i]["boxes"])
                all_gt_labels.append(targets[i]["labels"])

    all_scores_t = torch.stack(all_scores)   # [N_val, 300]
    print(f"\n=== Score Distribution (across {len(all_scores)} val images) ===")
    flat = all_scores_t.flatten()
    for p in [50, 90, 95, 99, 99.9]:
        print(f"  p{p:5.1f}: {torch.quantile(flat, p/100):.6f}")
    print(f"  max:   {flat.max():.6f}")
    print(f"  mean:  {flat.mean():.6f}")

    top_k = all_scores_t.topk(25, dim=-1).values
    print(f"\n  top-25 mean per image: {top_k.mean():.6f}")
    print(f"  top-1  mean per image: {all_scores_t.max(dim=-1).values.mean():.6f}")

    # Threshold sweep
    try:
        from torchmetrics.detection import MeanAveragePrecision
    except ImportError:
        print("torchmetrics not available — skipping MAP sweep")
        return

    thresholds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3]
    print(f"\n=== MAP Threshold Sweep ===")
    print(f"{'threshold':>12}  {'preds/img':>10}  {'map@0.5':>10}  {'map@0.5:0.95':>14}")

    for thresh in thresholds:
        metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
        preds_per_img = []
        for i in range(len(all_scores)):
            s = all_scores[i]
            keep = s > thresh
            preds_per_img.append(keep.sum().item())
            metric.update(
                [{"boxes": all_pred_boxes[i][keep], "scores": s[keep],
                  "labels": torch.zeros(keep.sum(), dtype=torch.long)}],
                [{"boxes": all_gt_boxes[i], "labels": all_gt_labels[i]}],
            )
        result = metric.compute()
        print(f"  {thresh:>10.3f}  {np.mean(preds_per_img):>10.1f}  "
              f"{result['map_50'].item():>10.4f}  {result['map'].item():>14.4f}")


if __name__ == "__main__":
    main()
