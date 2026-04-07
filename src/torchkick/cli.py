"""
Command-line interface for torchkick.

Provides subcommands for common operations:
    - analyze: Run full match analysis pipeline
    - train: Train detection models
    - prelabel: Run pre-labeling pipeline for CVAT annotation
    - download: Download model weights and datasets
    - dataset: Download SoccerNet datasets

Usage:
    torchkick analyze --video match.mp4
    torchkick train yolo --data soccernet/tracking/train.zip
    torchkick prelabel --help
    torchkick download --help

Example:
    $ torchkick analyze --video match.mp4 --model yolo --duration 60
    $ torchkick train yolo --epochs 50 --batch-size 256
    $ torchkick prelabel --video match.mp4 --output annotations.json
    $ torchkick download --weights player-detector
"""

from __future__ import annotations

import click


@click.group()
@click.version_option(package_name="torchkick")
def main() -> None:
    """
    torchkick: Computer vision toolkit for football/soccer video analysis.

    Run 'torchkick COMMAND --help' for more information on a command.
    """
    pass


@main.command()
@click.option(
    "--video",
    "-v",
    type=click.Path(exists=True),
    required=True,
    help="Path to input video file.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output path for annotations. Defaults to video name + .json.",
)
@click.option(
    "--project-id",
    type=int,
    default=None,
    help="CVAT project ID to upload to (optional).",
)
@click.option(
    "--task-name",
    type=str,
    default=None,
    help="CVAT task name (defaults to video filename).",
)
@click.option(
    "--max-duration",
    type=float,
    default=None,
    help="Maximum video duration to process in seconds.",
)
@click.option(
    "--skip-upload",
    is_flag=True,
    help="Generate annotations locally without uploading to CVAT.",
)
def prelabel(
    video: str,
    output: str | None,
    project_id: int | None,
    task_name: str | None,
    max_duration: float | None,
    skip_upload: bool,
) -> None:
    """
    Run pre-labeling pipeline for CVAT annotation.

    Processes a video file, runs detection and tracking models, and either
    uploads results to CVAT or saves locally as annotation files.

    Example:
        $ torchkick prelabel -v match.mp4 --project-id 123
        $ torchkick prelabel -v match.mp4 -o annotations.json --skip-upload
    """
    click.echo(f"Processing video: {video}")
    click.echo(f"Max duration: {max_duration or 'full video'}")

    if skip_upload:
        output_path = output or video.rsplit(".", 1)[0] + "_annotations.json"
        click.echo(f"Saving annotations to: {output_path}")
        # TODO: Import and run prelabel pipeline when annotation module is ready
        click.echo("Pre-labeling pipeline not yet implemented in new structure.")
    else:
        if project_id is None:
            raise click.UsageError("--project-id is required for CVAT upload")
        click.echo(f"Uploading to CVAT project: {project_id}")
        click.echo("CVAT upload not yet implemented in new structure.")


@main.command()
@click.option(
    "--weights",
    "-w",
    type=click.Choice(["player-detector", "pitch-lines", "all"]),
    required=True,
    help="Which model weights to download.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default=None,
    help="Directory to save weights. Defaults to ~/.torchkick/weights/.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing weights if present.",
)
def download(
    weights: str,
    output_dir: str | None,
    force: bool,
) -> None:
    """
    Download model weights.

    Downloads pre-trained model weights for player detection, pitch line
    detection, or both.

    Example:
        $ torchkick download -w player-detector
        $ torchkick download -w all -o ./weights/
    """
    from pathlib import Path

    if output_dir is None:
        output_dir = str(Path.home() / ".torchkick" / "weights")

    click.echo(f"Downloading {weights} weights to: {output_dir}")

    # TODO: Implement actual download logic when download module is ready
    if weights in ("player-detector", "all"):
        click.echo("  - Player detector: not yet implemented")
    if weights in ("pitch-lines", "all"):
        click.echo("  - Pitch lines: not yet implemented")

    click.echo("Download functionality coming soon!")


@main.command()
@click.option(
    "--dataset",
    "-d",
    type=click.Choice(["tracking", "calibration", "all", "roboflow-players", "roboflow-field", "roboflow"]),
    required=True,
    help="Which dataset to download.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="./data/soccernet/",
    help="Directory to save dataset files.",
)
@click.option(
    "--splits",
    type=str,
    default="train,test",
    help="Comma-separated list of splits to download (train,test,challenge). SoccerNet only.",
)
@click.option(
    "--workspace",
    type=str,
    default=None,
    help="Roboflow workspace slug (roboflow.com/<workspace>/<project>). Required for --dataset roboflow.",
)
@click.option(
    "--project",
    type=str,
    default=None,
    help="Roboflow project slug. Required for --dataset roboflow.",
)
@click.option(
    "--version",
    type=int,
    default=1,
    help="Roboflow dataset version number (default 1).",
)
@click.option(
    "--format",
    "fmt",
    type=str,
    default="yolov8",
    help="Roboflow export format: yolov8 (default) or coco.",
)
@click.option(
    "--api-key",
    type=str,
    default=None,
    help="Roboflow API key. Falls back to ROBOFLOW_API_KEY in .env.",
)
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
    Download training datasets.

    Downloads SoccerNet tracking/calibration data, or any Roboflow Universe
    dataset by workspace and project slug (visible in the dataset URL).

    Example:
        $ torchkick dataset -d tracking -o ./data/
        $ torchkick dataset -d roboflow-players -o data/roboflow/players/
        $ torchkick dataset -d roboflow-field   -o data/roboflow/field/
        $ torchkick dataset -d roboflow \\
              --workspace my-workspace --project my-project --version 3 \\
              -o data/custom/
    """
    from pathlib import Path

    if dataset in ("tracking", "calibration", "all"):
        split_list = [s.strip() for s in splits.split(",")]
        click.echo(f"Downloading SoccerNet {dataset} dataset")
        click.echo(f"  Splits: {split_list}")
        click.echo(f"  Output: {output_dir}")

        try:
            from torchkick.soccernet import download_tracking_data, download_pitch_calibration

            if dataset in ("tracking", "all"):
                tracking_dir = str(Path(output_dir) / "tracking")
                click.echo(f"Downloading tracking data to {tracking_dir}...")
                download_tracking_data(tracking_dir, splits=split_list, include_2023=False)  # type: ignore
                click.echo("  ✓ Tracking data downloaded")

            if dataset in ("calibration", "all"):
                calibration_dir = str(Path(output_dir) / "calibration")
                click.echo(f"Downloading calibration data to {calibration_dir}...")
                download_pitch_calibration(calibration_dir, splits=split_list)  # type: ignore
                click.echo("  ✓ Calibration data downloaded")

        except ImportError:
            raise click.UsageError(
                "SoccerNet package is required for dataset downloads. " "Install with: pip install torchkick[soccernet]"
            )

    elif dataset == "roboflow-players":
        from torchkick.soccernet import download_roboflow_players

        click.echo(f"Downloading Roboflow player detection dataset to {output_dir}...")
        path = download_roboflow_players(output_dir, api_key=api_key)
        click.echo(f"  ✓ Downloaded to {path}")

    elif dataset == "roboflow-field":
        from torchkick.soccernet import download_roboflow_field_keypoints

        click.echo(f"Downloading Roboflow field keypoints dataset to {output_dir}...")
        path = download_roboflow_field_keypoints(output_dir, api_key=api_key)
        click.echo(f"  ✓ Downloaded to {path}")

    elif dataset == "roboflow":
        if not workspace or not project:
            raise click.UsageError(
                "--workspace and --project are required for Roboflow downloads.\n"
                "Find them in the dataset URL: roboflow.com/<workspace>/<project>"
            )
        from torchkick.soccernet import download_roboflow_dataset

        click.echo(f"Downloading {workspace}/{project} v{version} ({fmt}) to {output_dir}...")
        path = download_roboflow_dataset(
            workspace=workspace,
            project=project,
            version=version,
            output_dir=output_dir,
            fmt=fmt,
            api_key=api_key,
        )
        click.echo(f"  ✓ Dataset downloaded to {path}")


# =============================================================================
# ANALYZE COMMAND - Full match analysis pipeline
# =============================================================================


@main.command()
@click.option(
    "--video",
    "-v",
    type=click.Path(exists=True),
    required=True,
    help="Path to input video file.",
)
@click.option(
    "--model",
    "-m",
    type=str,
    default=None,
    help="Path to detection model weights. Defaults based on --model-type.",
)
@click.option(
    "--model-type",
    type=click.Choice(["yolo", "fcnn", "rtdetr", "rfdetr", "sam3_mlx", "sam3_pytorch"]),
    default="rfdetr",
    help="Detection model type. rfdetr (DINOv2, AP50 73.6) is the recommended default.",
)
@click.option(
    "--duration",
    "-d",
    type=float,
    default=None,
    help="Maximum duration to process in seconds.",
)
@click.option(
    "--homography-interval",
    type=int,
    default=1,
    help="Frames between homography updates (1=every frame).",
)
@click.option(
    "--no-overlay",
    "no_overlay",
    is_flag=True,
    default=False,
    help="Disable pitch line overlay on video.",
)
@click.option(
    "--no-dominance",
    "no_dominance",
    is_flag=True,
    default=False,
    help="Disable space control heatmap.",
)
@click.option(
    "--reid-weights",
    type=click.Path(),
    default=None,
    help="Path to ReID student checkpoint for appearance-guided tracking.",
)
@click.option(
    "--reid-interval",
    type=int,
    default=5,
    help="Frames between ReID embedding updates (default 5).",
)
@click.option(
    "--pitch-weights",
    type=click.Path(),
    default=None,
    help="Path to pitch keypoint model weights for homography estimation.",
)
@click.option(
    "--pitch-detector-type",
    type=click.Choice(["yolo", "vitpose"]),
    default="yolo",
    help="Pitch keypoint detector backend: 'yolo' (faster) or 'vitpose' (more accurate).",
)
@click.option(
    "--conf-threshold",
    type=float,
    default=0.3,
    help="Detection confidence threshold (default 0.3; lower for early-stage checkpoints).",
)
@click.option(
    "--pose",
    "enable_pose",
    is_flag=True,
    default=False,
    help="Enable body pose estimation and 3D lifting.",
)
@click.option(
    "--pose-weights",
    type=str,
    default=None,
    help="ViTPose HF model ID or local path (default: usyd-community/vitpose-base-simple).",
)
def analyze(
    video: str,
    model: str | None,
    model_type: str,
    duration: float | None,
    homography_interval: int,
    no_overlay: bool,
    no_dominance: bool,
    reid_weights: str | None,
    reid_interval: int,
    pitch_weights: str | None,
    pitch_detector_type: str,
    conf_threshold: float,
    enable_pose: bool,
    pose_weights: str | None,
) -> None:
    """
    Run full match analysis pipeline.

    Performs player detection, tracking, pitch projection, team classification,
    and outputs a visualization video with 2D pitch view.

    RF-DETR (DINOv2 backbone, AP50 73.6) is the default and recommended
    detector. When no --reid-weights are given, SigLIP zero-shot clustering
    is used for team assignment automatically.

    Example:
        $ torchkick analyze -v match.mp4
        $ torchkick analyze -v match.mp4 --model-type yolo --duration 60
        $ torchkick analyze -v match.mp4 --reid-weights weights/reid/student.pt
    """
    from torchkick.inference import run_analysis

    click.echo(f"Running match analysis on: {video}")
    click.echo(f"  Model type: {model_type}")
    click.echo(f"  Duration: {duration or 'full video'}")

    output_path = run_analysis(
        video_path=video,
        model_path=model,
        model_type=model_type,
        duration=duration,
        homography_interval=homography_interval,
        draw_overlay=not no_overlay,
        draw_dominance=not no_dominance,
        reid_weights=reid_weights,
        reid_interval=reid_interval,
        pitch_weights=pitch_weights,
        pitch_detector_type=pitch_detector_type,
        conf_threshold=conf_threshold,
        enable_pose=enable_pose,
        pose_weights=pose_weights,
    )

    click.echo(f"Analysis complete! Output: {output_path}")


# =============================================================================
# TRAIN COMMAND GROUP - Model training
# =============================================================================


@main.group()
def train() -> None:
    """
    Train models for football analysis.

    Example:
        $ torchkick train detection --soccernet data/sn/train.zip --roboflow-json data/rf/train/_ann.json --roboflow-images data/rf/train/
        $ torchkick train yolo --data soccernet/tracking/train.zip
        $ torchkick train keypoints --data calibration/train.zip
        $ torchkick train reid --stage supervised --data-dir data/reid/
        $ torchkick train distill --teacher-weights weights/reid/reid_supervised_best.pth --data-dir data/reid/
    """
    pass


@train.command("detection")
@click.option(
    "--soccernet",
    type=click.Path(exists=True),
    default=None,
    help="SoccerNet tracking ZIP file (e.g. data/soccernet/tracking/train.zip).",
)
@click.option(
    "--soccernet-dir",
    type=click.Path(exists=True),
    default=None,
    help="Pre-extracted SoccerNet directory (faster than zip). Extract with: unzip train.zip -d data/soccernet_extracted/",
)
@click.option(
    "--roboflow-json",
    type=click.Path(exists=True),
    default=None,
    help="Roboflow COCO JSON annotations (e.g. data/roboflow/train/_annotations.coco.json).",
)
@click.option(
    "--roboflow-images",
    type=click.Path(exists=True),
    default=None,
    help="Directory containing Roboflow images (required when --roboflow-json is given).",
)
@click.option(
    "--cvat-json",
    type=click.Path(exists=True),
    default=None,
    help="CVAT COCO 1.0 export JSON (e.g. data/custom/annotations.json).",
)
@click.option(
    "--cvat-images",
    type=click.Path(exists=True),
    default=None,
    help="Directory containing CVAT images (required when --cvat-json is given).",
)
@click.option(
    "--person-ball",
    is_flag=True,
    default=True,
    help="Remap all person categories (player/goalkeeper/referee) → 0, ball → 1. Recommended.",
)
@click.option("--epochs", "-e", type=int, default=300)
@click.option("--batch-size", "-b", type=int, default=8)
@click.option(
    "--grad-accumulation",
    type=int,
    default=2,
    help="Gradient accumulation steps. Effective batch = batch_size * grad_accumulation.",
)
@click.option("--lr", type=float, default=1e-5)
@click.option(
    "--lr-backbone-scale",
    type=float,
    default=0.1,
    help="Backbone LR multiplier (backbone_lr = lr * scale). 0.1 = 10x lower than decoder. Use 0.0 to freeze.",
)
@click.option(
    "--num-classes",
    type=int,
    default=1,
    help="Number of output classes. Default 1 = player only (SoccerNet has no per-class annotation).",
)
@click.option("--save-dir", type=str, default="weights/detection/")
@click.option("--no-compile", is_flag=True, help="Disable torch.compile.")
@click.option("--fsdp", is_flag=True, help="Enable FSDP for multi-GPU training.")
@click.option("--wandb-project", type=str, default=None)
@click.option(
    "--focal-gamma", type=float, default=3.0, help="VFL focal gamma (default 3.0; raise to suppress flat scores)."
)
@click.option(
    "--resume", type=click.Path(exists=True), default=None, help="Resume training from a saved checkpoint (.pth)."
)
@click.option("--cls-head-lr-scale", type=float, default=20.0, help="cls_head LR multiplier (peak LR = lr * scale).")
@click.option(
    "--cls-head-decay-epochs", type=int, default=30, help="Epochs over which cls_head LR cosine-decays to 1%% of peak."
)
@click.option(
    "--backbone",
    type=click.Choice(["r50", "r101", "swin_t"]),
    default="r101",
    help="Backbone: r50/r101 (COCO pretrained ResNet) or swin_t (ImageNet-22k Swin-T via timm).",
)
@click.option("--mlp-head", is_flag=True, default=False, help="Replace linear cls heads with 2-layer MLP + LayerNorm.")
@click.option(
    "--backbone-phase1-epochs",
    type=int,
    default=5,
    help="Epochs to hold backbone at near-zero LR before ramping. Protects pretrained features during head warm-up.",
)
@click.option(
    "--backbone-ramp-epochs",
    type=int,
    default=5,
    help="Epochs to linearly ramp backbone LR from near-zero to peak after phase1.",
)
def train_detection_cmd(
    soccernet: str | None,
    soccernet_dir: str | None,
    roboflow_json: str | None,
    roboflow_images: str | None,
    cvat_json: str | None,
    cvat_images: str | None,
    person_ball: bool,
    epochs: int,
    batch_size: int,
    grad_accumulation: int,
    lr: float,
    lr_backbone_scale: float,
    num_classes: int,
    save_dir: str,
    no_compile: bool,
    fsdp: bool,
    wandb_project: str | None,
    focal_gamma: float,
    resume: str | None,
    cls_head_lr_scale: float,
    cls_head_decay_epochs: int,
    backbone: str,
    mlp_head: bool,
    backbone_phase1_epochs: int,
    backbone_ramp_epochs: int,
) -> None:
    """
    Train RT-DETR player detector from any combination of data sources.

    Accepts any non-empty subset of SoccerNet, Roboflow, and CVAT data.
    All specified sources are merged at training time — no manual dataset
    merging required.

    Data source formats:
        SoccerNet : ZIP file from `torchkick dataset -d tracking`
        Roboflow  : COCO JSON export (Roboflow → Export → COCO JSON)
        CVAT      : COCO 1.0 export  (CVAT → Export dataset → COCO 1.0)

    Example:
        $ torchkick train detection --soccernet data/soccernet/tracking/train.zip

        $ torchkick train detection \\
              --roboflow-json data/roboflow/train/_annotations.coco.json \\
              --roboflow-images data/roboflow/train/

        $ torchkick train detection \\
              --soccernet data/soccernet/tracking/train.zip \\
              --roboflow-json data/roboflow/train/_annotations.coco.json \\
              --roboflow-images data/roboflow/train/ \\
              --cvat-json data/custom/annotations.json \\
              --cvat-images data/custom/images/
    """
    from torchkick.training import train_detection

    # Validate paired arguments
    if roboflow_json and not roboflow_images:
        raise click.UsageError("--roboflow-images is required when --roboflow-json is given.")
    if cvat_json and not cvat_images:
        raise click.UsageError("--cvat-images is required when --cvat-json is given.")

    # Build data_config from whichever sources were provided
    data_config = []
    if soccernet:
        data_config.append({"type": "soccernet_zip", "path": soccernet})
        click.echo(f"  + SoccerNet ZIP: {soccernet}")
    if soccernet_dir:
        data_config.append({"type": "soccernet_dir", "path": soccernet_dir})
        click.echo(f"  + SoccerNet dir: {soccernet_dir}")
    # Map all person sub-types to COCO class 0 (person); drop ball and supercategory.
    # Ball detection is handled by a separate YOLOv11n model — mixing them causes
    # partial-label noise (SoccerNet has no ball annotations).
    label_map = {0: -1, 1: -1, 2: 0, 3: 0, 4: 0} if person_ball else None
    if roboflow_json:
        src = {"type": "coco_json", "path": roboflow_json, "images_dir": roboflow_images}
        if label_map:
            src["label_map"] = label_map
        data_config.append(src)
        click.echo(f"  + Roboflow:  {roboflow_json}" + (" (person+ball remapping)" if label_map else ""))
    if cvat_json:
        src = {"type": "coco_json", "path": cvat_json, "images_dir": cvat_images}
        if label_map:
            src["label_map"] = label_map
        data_config.append(src)
        click.echo(f"  + CVAT:      {cvat_json}" + (" (person+ball remapping)" if label_map else ""))

    if not data_config:
        raise click.UsageError(
            "At least one data source is required. Use --soccernet, --soccernet-dir, --roboflow-json, or --cvat-json."
        )

    click.echo(f"Training RT-DETR detection model ({len(data_config)} source(s))")

    weights = train_detection(
        data_config=data_config,
        num_labels=num_classes,
        epochs=epochs,
        batch_size=batch_size,
        grad_accumulation=grad_accumulation,
        learning_rate=lr,
        lr_backbone_scale=lr_backbone_scale,
        compile_model=not no_compile,
        use_fsdp=fsdp,
        save_dir=save_dir,
        wandb_project=wandb_project,
        focal_gamma=focal_gamma,
        resume_from=resume,
        cls_head_lr_scale=cls_head_lr_scale,
        cls_head_decay_epochs=cls_head_decay_epochs,
        backbone=backbone,
        use_mlp_head=mlp_head,
        backbone_phase1_epochs=backbone_phase1_epochs,
        backbone_ramp_epochs=backbone_ramp_epochs,
    )
    click.echo(f"Training complete! Best model: {weights}")


@train.command("yolo")
@click.option("--data", "-d", type=click.Path(), default=None, help="Path to SoccerNet zip or YOLO dataset directory.")
@click.option("--soccernet-dir", type=click.Path(exists=True), default=None, help="Pre-extracted SoccerNet directory.")
@click.option("--epochs", "-e", type=int, default=100)
@click.option("--batch-size", "-b", type=int, default=32)
@click.option("--colors", is_flag=True, help="Train with jersey color classification (7 classes).")
@click.option(
    "--base-model",
    type=str,
    default="yolo11l.pt",
    help="Base YOLO model: yolo11n/s/m/l/x.pt. Default yolo11l (25M params, best accuracy/speed tradeoff).",
)
@click.option(
    "--save-dir", type=str, default="/workspace/weights/yolo_detection/", help="Output directory for weights."
)
@click.option(
    "--frame-stride",
    type=int,
    default=1,
    help="Keep every Nth frame during dataset conversion (default 1 = all frames). Use 5 to cut disk usage ~5x.",
)
def train_yolo_cmd(
    data: str | None,
    soccernet_dir: str | None,
    epochs: int,
    batch_size: int,
    colors: bool,
    base_model: str,
    save_dir: str,
    frame_stride: int,
) -> None:
    """
    Train YOLO player detector on SoccerNet data.

    Example:
        $ torchkick train yolo --soccernet-dir /tmp/soccernet_extracted/ --epochs 100
        $ torchkick train yolo --data tracking/train.zip --base-model yolo11l.pt
    """
    from torchkick.training import train_yolo

    click.echo(f"Training YOLO ({base_model}) player detector")
    click.echo(f"  Epochs: {epochs} | Batch: {batch_size}")

    weights = train_yolo(
        data_zip=data if data and data.endswith(".zip") else None,
        data_dir=data if data and not data.endswith(".zip") else None,
        soccernet_dir=soccernet_dir,
        epochs=epochs,
        batch_size=batch_size,
        use_colors=colors,
        base_model=base_model,
        project=save_dir,
        frame_stride=frame_stride,
    )

    click.echo(f"Training complete! Best model: {weights}")


@train.command("yolo-keypoints")
@click.option("--data", "-d", type=click.Path(exists=True), required=True, help="Path to YOLO-pose dataset YAML.")
@click.option("--epochs", "-e", type=int, default=300)
@click.option("--imgsz", type=int, default=320, help="Input image size.")
@click.option("--base-model", type=str, default="yolo11n-pose.pt", help="Base YOLO-pose model.")
@click.option("--save-dir", type=str, default="weights/keypoints/")
def train_yolo_keypoints_cmd(
    data: str,
    epochs: int,
    imgsz: int,
    base_model: str,
    save_dir: str,
) -> None:
    """
    Train YOLO-pose pitch keypoint detector (mosaic=0.0).

    Faster than ViTPose (~3ms/frame at 320×320). Use mosaic=0.0 to avoid
    spatial landmark shuffling that degrades keypoint AP.

    Example:
        $ torchkick train yolo-keypoints --data pitch.yaml --epochs 100
    """
    from torchkick.training.train_yolo_detection import train_yolo_keypoints

    click.echo(f"Training YOLO-pose keypoints (mosaic=0.0)")
    weights = train_yolo_keypoints(
        data_yaml=data,
        base_model=base_model,
        epochs=epochs,
        imgsz=imgsz,
        save_dir=save_dir,
    )
    click.echo(f"Training complete! Best model: {weights}")


@train.command("keypoints")
@click.option("--data", "-d", type=click.Path(exists=True), required=True, help="Path to SoccerNet calibration zip.")
@click.option("--epochs", "-e", type=int, default=300)
@click.option("--batch-size", "-b", type=int, default=8)
@click.option("--lr", type=float, default=5e-4, help="Peak learning rate.")
@click.option("--save-dir", type=str, default="weights/keypoints/")
@click.option("--model-variant", type=click.Choice(["ViTPose-L", "ViTPose-B"]), default="ViTPose-L")
@click.option("--no-compile", is_flag=True, help="Disable torch.compile.")
@click.option("--wandb-project", type=str, default=None)
def train_keypoints_cmd(
    data: str,
    epochs: int,
    batch_size: int,
    lr: float,
    save_dir: str,
    model_variant: str,
    no_compile: bool,
    wandb_project: str | None,
) -> None:
    """
    Train ViTPose-L pitch keypoint detector.

    Detects 29 pitch landmarks from SoccerNet calibration data.

    Example:
        $ torchkick train keypoints --data data/calibration/train.zip --epochs 100
    """
    from torchkick.training import train_keypoints

    click.echo(f"Training ViTPose ({model_variant}) keypoint detector")
    weights = train_keypoints(
        data_zip=data,
        model_variant=model_variant,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        compile_model=not no_compile,
        save_dir=save_dir,
        wandb_project=wandb_project,
    )
    click.echo(f"Training complete! Best model: {weights}")


@train.command("reid")
@click.option(
    "--data-dir",
    "-d",
    type=click.Path(exists=True),
    required=True,
    help="Directory with crop sub-folders by class (supervised) or track ID (ssl).",
)
@click.option(
    "--stage",
    type=click.Choice(["supervised", "ssl"]),
    default="supervised",
    help="Training stage: supervised ArcFace or BYOL tracklet SSL.",
)
@click.option("--epochs", "-e", type=int, default=30)
@click.option("--batch-size", "-b", type=int, default=64)
@click.option("--lr", type=float, default=3e-4)
@click.option("--lora-rank", type=int, default=16, help="LoRA adapter rank.")
@click.option("--num-classes", type=int, default=3, help="Identity classes (3 = home/away/ref).")
@click.option("--save-dir", type=str, default="weights/reid/")
@click.option("--no-compile", is_flag=True)
@click.option("--fsdp", is_flag=True, help="Enable FSDP for multi-GPU training.")
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
    fsdp: bool,
    wandb_project: str | None,
) -> None:
    """
    Train DINOv2 + LoRA + ArcFace ReID model.

    Stage 'supervised': ArcFace metric learning on labelled crops.
    Stage 'ssl': BYOL tracklet self-supervised learning.

    Example:
        $ torchkick train reid --stage supervised --data-dir data/reid/ --epochs 30
        $ torchkick train reid --stage ssl       --data-dir data/reid/ --epochs 20
    """
    from torchkick.training import train_reid

    click.echo(f"Training ReID ({stage} stage)")
    weights = train_reid(
        data_dir=data_dir,
        stage=stage,
        num_classes=num_classes,
        lora_rank=lora_rank,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        compile_model=not no_compile,
        use_fsdp=fsdp,
        save_dir=save_dir,
        wandb_project=wandb_project,
    )
    click.echo(f"Training complete! Best model: {weights}")


@train.command("reid-video")
@click.option("--video", "-v", type=click.Path(exists=True), required=True, help="Input video path.")
@click.option(
    "--yolo-weights",
    type=click.Path(exists=True),
    required=True,
    help="YOLO detection weights for crop extraction.",
)
@click.option("--save-dir", type=str, default="weights/reid/")
@click.option("--calibration-duration", type=float, default=120.0, help="Seconds to sample from (default 120).")
@click.option("--n-sample-frames", type=int, default=60, help="Number of frames to sample.")
@click.option("--epochs", "-e", type=int, default=20)
@click.option("--batch-size", "-b", type=int, default=64)
@click.option("--lora-rank", type=int, default=16)
@click.option("--keep-tmp", is_flag=True, help="Keep the temporary pseudo-labeled crop directory.")
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

    Runs YOLO on sampled frames, clusters crops into home/away/ref with
    SigLIP + k-means, then fine-tunes DINOv2+LoRA+ArcFace on the
    pseudo-labels.  No manually labeled data required.

    Example:
        $ torchkick train reid-video --video match.mp4 --yolo-weights best.pt --epochs 20
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
    click.echo(f"Training complete! Best model: {weights}")


@train.command("distill")
@click.option(
    "--teacher-weights",
    type=click.Path(exists=True),
    required=True,
    help="Path to supervised ReID checkpoint (reid_supervised_best.pth).",
)
@click.option(
    "--data-dir",
    "-d",
    type=click.Path(exists=True),
    required=True,
    help="ReID crops directory (same as used for teacher).",
)
@click.option("--epochs", "-e", type=int, default=30)
@click.option("--batch-size", "-b", type=int, default=128)
@click.option("--lr", type=float, default=3e-4)
@click.option("--temperature", type=float, default=4.0, help="Softmax temperature for KL distillation loss.")
@click.option("--save-dir", type=str, default="weights/reid/")
@click.option("--no-compile", is_flag=True)
@click.option("--wandb-project", type=str, default=None)
def train_distill_cmd(
    teacher_weights: str,
    data_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    temperature: float,
    save_dir: str,
    no_compile: bool,
    wandb_project: str | None,
) -> None:
    """
    Distil DINOv2-Large ReID teacher → DINOv2-Small student.

    Produces a 384-dim student model for real-time inference (~3ms/frame).

    Example:
        $ torchkick train distill \\
              --teacher-weights weights/reid/reid_supervised_best.pth \\
              --data-dir data/reid/
    """
    from torchkick.training import train_distill

    click.echo("Distilling ViT-L/14 → ViT-S/8")
    weights = train_distill(
        teacher_weights=teacher_weights,
        data_dir=data_dir,
        temperature=temperature,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        compile_model=not no_compile,
        save_dir=save_dir,
        wandb_project=wandb_project,
    )
    click.echo(f"Distillation complete! Student: {weights}")


# =============================================================================
# LABEL COMMAND GROUP - Automated annotation pipeline
# =============================================================================


@main.group()
def label() -> None:
    """
    Automated annotation tools.

    Run pre-labeling pipelines that generate CVAT-ready annotations
    from raw video using foundation models.

    Example:
        $ torchkick label grounded-sam --video match.mp4 --project-id 1
        $ torchkick label active-learning --video-dir data/videos/ --project-id 1
    """
    pass



@train.command("body-pose")
@click.option("--data", "-d", type=click.Path(exists=True), required=True, help="Path to downloaded FIFA dataset root.")
@click.option("--epochs", "-e", type=int, default=30)
@click.option("--batch-size", "-b", type=int, default=16)
@click.option("--lr", type=float, default=1e-4, help="Peak learning rate.")
@click.option(
    "--model-id", type=str, default="usyd-community/vitpose-base-simple", help="ViTPose HF model ID or local path."
)
@click.option("--save-dir", type=str, default="weights/body_pose/")
@click.option("--wandb-project", type=str, default=None)
@click.option("--max-samples", type=int, default=None, help="Cap dataset size for smoke tests.")
def train_body_pose_cmd(
    data: str,
    epochs: int,
    batch_size: int,
    lr: float,
    model_id: str,
    save_dir: str,
    wandb_project: str | None,
    max_samples: int | None,
) -> None:
    """
    Fine-tune ViTPose on FIFA body pose data (COCO-17 joints).

    Downloads the FIFA dataset first:
        huggingface-cli download tijiang13/FIFA-Skeletal-Tracking-Light-2026
            --repo-type dataset --local-dir data/fifa/

    Example:
        $ torchkick train body-pose --data data/fifa/ --epochs 30
    """
    from torchkick.training import train_body_pose

    click.echo(f"Fine-tuning ViTPose on {data}")
    weights = train_body_pose(
        data_dir=data,
        model_id=model_id,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        save_dir=save_dir,
        wandb_project=wandb_project,
        max_samples=max_samples,
    )
    click.echo(f"Training complete! Best model: {weights}")


@train.command("pose-lifter")
@click.option("--data", "-d", type=click.Path(exists=True), required=True, help="Path to downloaded FIFA dataset root.")
@click.option("--epochs", "-e", type=int, default=50)
@click.option("--batch-size", "-b", type=int, default=512)
@click.option("--lr", type=float, default=1e-3)
@click.option("--hidden-dim", type=int, default=1024)
@click.option("--n-blocks", type=int, default=4)
@click.option("--save-dir", type=str, default="weights/lifter/")
@click.option("--wandb-project", type=str, default=None)
@click.option("--max-samples", type=int, default=None, help="Cap dataset size for smoke tests.")
def train_pose_lifter_cmd(
    data: str,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_dim: int,
    n_blocks: int,
    save_dir: str,
    wandb_project: str | None,
    max_samples: int | None,
) -> None:
    """
    Train 2D→3D pose lifting MLP on FIFA paired pose data.

    Trains a residual MLP on Body25→COCO-17 remapped 2D/3D pairs.
    Reported metric: MPJPE (mm) on visible joints.

    Example:
        $ torchkick train pose-lifter --data data/fifa/ --epochs 50
    """
    from torchkick.training import train_pose_lifter

    click.echo(f"Training pose lifter on {data}")
    weights = train_pose_lifter(
        data_dir=data,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        hidden_dim=hidden_dim,
        n_blocks=n_blocks,
        save_dir=save_dir,
        wandb_project=wandb_project,
        max_samples=max_samples,
    )
    click.echo(f"Training complete! Best model: {weights}")


if __name__ == "__main__":
    main()
