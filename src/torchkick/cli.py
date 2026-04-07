"""
Command-line interface for torchkick.

Commands:
    analyze  — Run YOLO detection + 2D pitch reconstruction on a video
    train    — Train detection/keypoint/ReID models
    dataset  — Download SoccerNet / Roboflow datasets
    download — Download pre-trained model weights

Example:
    $ torchkick analyze --video match.mp4 --yolo-weights best.pt
    $ torchkick train yolo --soccernet-dir /data/soccernet/ --epochs 100
    $ torchkick train pitch-heatmap --data /data/soccernet_kp_dataset --epochs 100
    $ torchkick train reid-video --video match.mp4 --yolo-weights best.pt
"""

from __future__ import annotations

import click


@click.group()
@click.version_option(package_name="torchkick")
def main() -> None:
    """torchkick: computer vision toolkit for football video analysis."""
    pass


# =============================================================================
# ANALYZE
# =============================================================================


@main.command()
@click.option("--video", "-v", type=click.Path(exists=True), required=True, help="Input video.")
@click.option("--yolo-weights", type=click.Path(exists=True), required=True, help="YOLO player detection weights.")
@click.option(
    "--pitch-weights", type=click.Path(), default=None, help="YOLO-pose pitch keypoint weights (2D projection)."
)
@click.option(
    "--reid-weights", type=click.Path(), default=None, help="ReID checkpoint. If omitted, SigLIP zero-shot is used."
)
@click.option("--duration", "-d", type=float, default=None, help="Max seconds to process.")
@click.option("--homography-interval", type=int, default=1, help="Frames between homography updates.")
@click.option("--reid-interval", type=int, default=5, help="Frames between embedding updates.")
@click.option("--conf", type=float, default=0.25, help="Detection confidence threshold.")
@click.option("--no-overlay", is_flag=True, help="Disable pitch line wireframe overlay.")
def analyze(
    video: str,
    yolo_weights: str,
    pitch_weights: str | None,
    reid_weights: str | None,
    duration: float | None,
    homography_interval: int,
    reid_interval: int,
    conf: float,
    no_overlay: bool,
) -> None:
    """
    Run match analysis: YOLO detection, team assignment, 2D pitch projection.

    Passes:
      0. Calibrate team centroids from first 2 min (random frame sample)
      1. YOLO + BotSORT tracking + pitch projection
      1.5. Re-link fragmented tracks via ReID
      2. Smooth trajectories
      3. Assign teams + infer goalkeepers
      4. Render annotated video + pitch minimap

    When --reid-weights is omitted, SigLIP zero-shot clustering is used
    automatically — no training required.

    Example:
        $ torchkick analyze -v match.mp4 --yolo-weights weights/best.pt
        $ torchkick analyze -v match.mp4 --yolo-weights best.pt \\
              --pitch-weights pitch/best.pt --reid-weights reid/student.pt
    """
    from torchkick.inference import run_analysis

    click.echo(f"Analyzing: {video}")
    output = run_analysis(
        video_path=video,
        yolo_weights=yolo_weights,
        pitch_weights=pitch_weights,
        reid_weights=reid_weights,
        duration=duration,
        homography_interval=homography_interval,
        reid_interval=reid_interval,
        conf=conf,
        draw_overlay=not no_overlay,
    )
    click.echo(f"Done → {output}")


# =============================================================================
# DATASET / DOWNLOAD
# =============================================================================


@main.command()
@click.option(
    "--dataset",
    "-d",
    type=click.Choice(["tracking", "calibration", "all", "roboflow-players", "roboflow-field", "roboflow"]),
    required=True,
)
@click.option("--output-dir", "-o", type=click.Path(), default="./data/soccernet/")
@click.option("--splits", type=str, default="train,test", help="Comma-separated splits (SoccerNet only).")
@click.option("--workspace", type=str, default=None, help="Roboflow workspace slug.")
@click.option("--project", type=str, default=None, help="Roboflow project slug.")
@click.option("--version", type=int, default=1)
@click.option("--format", "fmt", type=str, default="yolov8")
@click.option("--api-key", type=str, default=None)
def dataset(
    dataset: str,
    output_dir: str,
    splits: str,
    workspace: str | None,
    project: str | None,
    version: int,
    fmt: str,
    api_key: str | None,
) -> None:
    """
    Download SoccerNet or Roboflow datasets.

    Example:
        $ torchkick dataset -d tracking -o ./data/
        $ torchkick dataset -d roboflow --workspace myws --project myproj -o data/
    """
    from pathlib import Path

    if dataset in ("tracking", "calibration", "all"):
        split_list = [s.strip() for s in splits.split(",")]
        try:
            from torchkick.soccernet import download_tracking_data, download_pitch_calibration

            if dataset in ("tracking", "all"):
                download_tracking_data(str(Path(output_dir) / "tracking"), splits=split_list, include_2023=False)
                click.echo("Tracking data downloaded.")
            if dataset in ("calibration", "all"):
                download_pitch_calibration(str(Path(output_dir) / "calibration"), splits=split_list)
                click.echo("Calibration data downloaded.")
        except ImportError:
            raise click.UsageError("Install torchkick[soccernet] for dataset downloads.")

    elif dataset == "roboflow-players":
        from torchkick.soccernet import download_roboflow_players

        path = download_roboflow_players(output_dir, api_key=api_key)
        click.echo(f"Downloaded to {path}")

    elif dataset == "roboflow-field":
        from torchkick.soccernet import download_roboflow_field_keypoints

        path = download_roboflow_field_keypoints(output_dir, api_key=api_key)
        click.echo(f"Downloaded to {path}")

    elif dataset == "roboflow":
        if not workspace or not project:
            raise click.UsageError("--workspace and --project required for Roboflow downloads.")
        from torchkick.soccernet import download_roboflow_dataset

        path = download_roboflow_dataset(
            workspace=workspace,
            project=project,
            version=version,
            output_dir=output_dir,
            fmt=fmt,
            api_key=api_key,
        )
        click.echo(f"Downloaded to {path}")


@main.command()
@click.option("--weights", "-w", type=click.Choice(["player-detector", "pitch-lines", "all"]), required=True)
@click.option("--output-dir", "-o", type=click.Path(), default=None)
def download(weights: str, output_dir: str | None) -> None:
    """Download pre-trained model weights (coming soon)."""
    click.echo("Weight download not yet available. Train your own with `torchkick train`.")


# =============================================================================
# TRAIN
# =============================================================================


@main.group()
def train() -> None:
    """Train detection, keypoint, and ReID models."""
    pass


# ---- YOLO player detection --------------------------------------------------


@train.command("yolo")
@click.option("--data", "-d", type=click.Path(), default=None, help="SoccerNet zip or YOLO dataset directory.")
@click.option("--soccernet-dir", type=click.Path(exists=True), default=None, help="Pre-extracted SoccerNet directory.")
@click.option("--epochs", "-e", type=int, default=100)
@click.option("--batch-size", "-b", type=int, default=32)
@click.option("--base-model", type=str, default="yolo11l.pt", help="yolo11n/s/m/l/x.pt")
@click.option("--save-dir", type=str, default="/workspace/weights/yolo_detection/")
@click.option("--frame-stride", type=int, default=1, help="Keep every Nth frame (1=all). Use 5 to cut disk usage ~5x.")
def train_yolo_cmd(
    data: str | None,
    soccernet_dir: str | None,
    epochs: int,
    batch_size: int,
    base_model: str,
    save_dir: str,
    frame_stride: int,
) -> None:
    """
    Train YOLO player/ball detector on SoccerNet data.

    Example:
        $ torchkick train yolo --soccernet-dir /data/soccernet/ --epochs 100
        $ torchkick train yolo --data tracking/train.zip --base-model yolo11l.pt
    """
    from torchkick.training import train_yolo

    click.echo(f"Training YOLO ({base_model}) | epochs={epochs} | batch={batch_size}")
    weights = train_yolo(
        data_zip=data if data and data.endswith(".zip") else None,
        data_dir=data if data and not data.endswith(".zip") else None,
        soccernet_dir=soccernet_dir,
        epochs=epochs,
        batch_size=batch_size,
        base_model=base_model,
        project=save_dir,
        frame_stride=frame_stride,
    )
    click.echo(f"Done → {weights}")


# ---- YOLO-pose pitch keypoints ----------------------------------------------


# ---- DINOv2 pitch heatmap keypoints ----------------------------------------


@train.command("pitch-heatmap")
@click.option(
    "--data",
    "-d",
    type=click.Path(exists=True),
    default=None,
    help="Pre-converted dataset root (images/ + labels/ subdirs).",
)
@click.option(
    "--soccernet-calibration-dir",
    type=click.Path(exists=True),
    default=None,
    help="SoccerNet calibration dir (train.zip / valid.zip). Auto-converts to label format.",
)
@click.option("--epochs", "-e", type=int, default=100)
@click.option("--batch-size", "-b", type=int, default=8)
@click.option(
    "--backbone",
    type=click.Choice(["dinov2_vits14", "dinov2_vitb14"]),
    default="dinov2_vits14",
    help="DINOv2 variant (S=~23M, B=~88M params).",
)
@click.option("--base-model", type=click.Path(), default=None, help="Resume / fine-tune from existing checkpoint.")
@click.option(
    "--min-keypoints", type=int, default=6, help="Skip frames with fewer visible keypoints (filters close-ups)."
)
@click.option("--imgsz", type=int, default=560, help="Input image size (must be multiple of 14).")
@click.option("--save-dir", type=str, default="weights/pitch_heatmap/")
@click.option("--wandb-project", type=str, default=None)
@click.option("--no-compile", is_flag=True, help="Disable torch.compile.")
def train_pitch_heatmap_cmd(
    data: str | None,
    soccernet_calibration_dir: str | None,
    epochs: int,
    batch_size: int,
    backbone: str,
    base_model: str | None,
    min_keypoints: int,
    imgsz: int,
    save_dir: str,
    wandb_project: str | None,
    no_compile: bool,
) -> None:
    """
    Train DINOv2+heatmap pitch keypoint detector (32-keypoint schema).

    Uses a ViT-S/14 (or ViT-B/14) backbone with full end-to-end fine-tuning
    (differential LR: backbone 10× lower than decoder) and a CNN heatmap
    decoder.  Includes a pitch-presence head for broadcast-view filtering.

    Accepts either a pre-converted dataset directory (--data) or a raw
    SoccerNet calibration directory (--soccernet-calibration-dir) which is
    auto-converted before training starts.

    Example:
        $ torchkick train pitch-heatmap \\
              --data /workspace/weights/soccernet_kp_dataset \\
              --epochs 100 --batch-size 8 --wandb-project torchkick

        $ torchkick train pitch-heatmap \\
              --soccernet-calibration-dir /data/soccernet/calibration \\
              --epochs 100 --batch-size 8
    """
    from torchkick.training import train_pitch_heatmap

    if data is None and soccernet_calibration_dir is None:
        raise click.UsageError("Provide --data or --soccernet-calibration-dir.")

    click.echo(f"Training DINOv2 heatmap pitch detector ({backbone}) | " f"epochs={epochs} | batch={batch_size}")
    best = train_pitch_heatmap(
        data_dir=data,
        soccernet_calibration_dir=soccernet_calibration_dir,
        base_model=base_model,
        backbone_variant=backbone,
        epochs=epochs,
        batch_size=batch_size,
        min_keypoints=min_keypoints,
        imgsz=imgsz,
        compile_model=not no_compile,
        save_dir=save_dir,
        wandb_project=wandb_project,
    )
    click.echo(f"Done → {best}")


# ---- DINOv2 ReID (labelled crops) ------------------------------------------


@train.command("reid")
@click.option(
    "--data-dir",
    "-d",
    type=click.Path(exists=True),
    required=True,
    help="Crop sub-folders by class (0/1/2) or track ID.",
)
@click.option("--stage", type=click.Choice(["supervised", "ssl"]), default="supervised")
@click.option("--epochs", "-e", type=int, default=30)
@click.option("--batch-size", "-b", type=int, default=64)
@click.option("--lr", type=float, default=3e-4)
@click.option("--lora-rank", type=int, default=16)
@click.option("--num-classes", type=int, default=3)
@click.option("--save-dir", type=str, default="weights/reid/")
@click.option("--no-compile", is_flag=True)
@click.option("--wandb-project", type=str, default=None)
def train_reid_cmd(
    data_dir: str,
    stage: str,
    epochs: int,
    batch_size: int,
    lr: float,
    lora_rank: int,
    num_classes: int,
    save_dir: str,
    no_compile: bool,
    wandb_project: str | None,
) -> None:
    """
    Train DINOv2+LoRA+ArcFace ReID (supervised ArcFace or BYOL SSL).

    Example:
        $ torchkick train reid --data-dir data/reid/ --epochs 30
        $ torchkick train reid --stage ssl --data-dir data/tracklets/ --epochs 20
    """
    from torchkick.training import train_reid

    click.echo(f"Training ReID ({stage})")
    weights = train_reid(
        data_dir=data_dir,
        stage=stage,
        num_classes=num_classes,
        lora_rank=lora_rank,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        compile_model=not no_compile,
        save_dir=save_dir,
        wandb_project=wandb_project,
    )
    click.echo(f"Done → {weights}")


# ---- ReID from raw video (auto-labels via SigLIP clustering) ---------------


@train.command("reid-video")
@click.option("--video", "-v", type=click.Path(exists=True), required=True)
@click.option("--yolo-weights", type=click.Path(exists=True), required=True, help="YOLO weights for crop extraction.")
@click.option("--save-dir", type=str, default="weights/reid/")
@click.option("--calibration-duration", type=float, default=120.0, help="Seconds to sample from.")
@click.option("--n-sample-frames", type=int, default=60)
@click.option("--epochs", "-e", type=int, default=20)
@click.option("--batch-size", "-b", type=int, default=64)
@click.option("--lora-rank", type=int, default=16)
@click.option("--keep-tmp", is_flag=True, help="Keep temporary pseudo-labeled crops.")
def train_reid_video_cmd(
    video: str,
    yolo_weights: str,
    save_dir: str,
    calibration_duration: float,
    n_sample_frames: int,
    epochs: int,
    batch_size: int,
    lora_rank: int,
    keep_tmp: bool,
) -> None:
    """
    Auto-label player crops from a video and train DINOv2+ArcFace ReID.

    No manually labeled data required — SigLIP + k-means generates pseudo-labels.

    Example:
        $ torchkick train reid-video --video match.mp4 --yolo-weights best.pt
    """
    from torchkick.training import train_reid_from_video

    click.echo(f"Training ReID from video: {video}")
    weights = train_reid_from_video(
        video_path=video,
        yolo_weights=yolo_weights,
        save_dir=save_dir,
        calibration_duration=calibration_duration,
        n_sample_frames=n_sample_frames,
        epochs=epochs,
        batch_size=batch_size,
        lora_rank=lora_rank,
        keep_tmp=keep_tmp,
    )
    click.echo(f"Done → {weights}")


# ---- FIFA / future models (stubs) ------------------------------------------


@train.command("detection")
@click.argument("args", nargs=-1)
def train_detection_cmd(args) -> None:
    """[Not implemented] RT-DETR player detection training. Use `train yolo` instead."""
    raise click.UsageError("RT-DETR training not active. Use `torchkick train yolo` for player detection.")


@train.command("keypoints")
@click.argument("args", nargs=-1)
def train_keypoints_cmd(args) -> None:
    """[Removed] Use `train pitch-heatmap` for DINOv2+heatmap pitch keypoint detection."""
    raise click.UsageError("Use `torchkick train pitch-heatmap` for pitch keypoint detection.")


@train.command("distill")
@click.argument("args", nargs=-1)
def train_distill_cmd(args) -> None:
    """[Not implemented] Knowledge distillation ViT-L → ViT-S."""
    raise click.UsageError("Distillation training not yet implemented.")


@train.command("body-pose")
@click.argument("args", nargs=-1)
def train_body_pose_cmd(args) -> None:
    """[Not implemented] ViTPose fine-tuning on FIFA body pose data."""
    raise click.UsageError("Body pose training not yet implemented.")


@train.command("pose-lifter")
@click.argument("args", nargs=-1)
def train_pose_lifter_cmd(args) -> None:
    """[Not implemented] 2D→3D pose lifting MLP on FIFA paired data."""
    raise click.UsageError("Pose lifter training not yet implemented.")


if __name__ == "__main__":
    main()
