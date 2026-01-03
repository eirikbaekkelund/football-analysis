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
    type=click.Choice(["tracking", "calibration", "all"]),
    required=True,
    help="Which SoccerNet dataset to download.",
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
    help="Comma-separated list of splits to download (train,test,challenge).",
)
def dataset(
    dataset: str,
    output_dir: str,
    splits: str,
) -> None:
    """
    Download SoccerNet datasets.

    Downloads tracking or calibration data from SoccerNet. Requires
    the soccernet optional dependency.

    Example:
        $ torchkick dataset -d tracking -o ./data/
        $ torchkick dataset -d all --splits train,test
    """
    from pathlib import Path

    split_list = [s.strip() for s in splits.split(",")]
    click.echo(f"Downloading SoccerNet {dataset} dataset")
    click.echo(f"  Splits: {split_list}")
    click.echo(f"  Output: {output_dir}")

    try:
        from torchkick.soccernet import download_tracking_data, download_pitch_calibration

        if dataset in ("tracking", "all"):
            tracking_dir = str(Path(output_dir) / "tracking")
            click.echo(f"Downloading tracking data to {tracking_dir}...")
            download_tracking_data(tracking_dir, splits=split_list)  # type: ignore
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
    type=click.Choice(["yolo", "fcnn", "rtdetr"]),
    default="fcnn",
    help="Detection model type.",
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
    "--overlay",
    is_flag=False,
    help="Disable pitch line overlay on video.",
)
@click.option(
    "--dominance",
    is_flag=False,
    help="Disable space control heatmap.",
)
def analyze(
    video: str,
    model: str | None,
    model_type: str,
    duration: float | None,
    homography_interval: int,
    no_overlay: bool,
    no_dominance: bool,
) -> None:
    """
    Run full match analysis pipeline.

    Performs player detection, tracking, pitch projection, team classification,
    and outputs a visualization video with 2D pitch view.

    Example:
        $ torchkick analyze -v match.mp4
        $ torchkick analyze -v match.mp4 --model-type yolo --duration 60
        $ torchkick analyze -v match.mp4 --dominance
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
    )

    click.echo(f"Analysis complete! Output: {output_path}")


# =============================================================================
# TRAIN COMMAND GROUP - Model training
# =============================================================================


@main.group()
def train() -> None:
    """
    Train detection models.

    Subcommands for training various models on SoccerNet data.

    Example:
        $ torchkick train yolo --data soccernet/tracking/train.zip
        $ torchkick train fcnn --epochs 10
        $ torchkick train lines --data calibration_data/
    """
    pass


@train.command("yolo")
@click.option(
    "--data",
    "-d",
    type=click.Path(),
    default=None,
    help="Path to SoccerNet zip or YOLO dataset directory.",
)
@click.option(
    "--epochs",
    "-e",
    type=int,
    default=50,
    help="Number of training epochs.",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=256,
    help="Training batch size.",
)
@click.option(
    "--colors",
    is_flag=True,
    help="Train with jersey color classification (7 classes).",
)
@click.option(
    "--base-model",
    type=str,
    default="yolo11n.pt",
    help="Base YOLO model to finetune.",
)
def train_yolo_cmd(
    data: str | None,
    epochs: int,
    batch_size: int,
    colors: bool,
    base_model: str,
) -> None:
    """
    Train YOLO player detector.

    Trains YOLOv11 on SoccerNet tracking data for player detection.

    Example:
        $ torchkick train yolo --epochs 50 --batch-size 256
        $ torchkick train yolo --data tracking/train.zip --colors
    """
    from torchkick.training import train_yolo

    click.echo("Training YOLO player detector")
    click.echo(f"  Epochs: {epochs}")
    click.echo(f"  Batch size: {batch_size}")
    click.echo(f"  Color classification: {colors}")

    weights = train_yolo(
        data_zip=data if data and data.endswith(".zip") else None,
        data_dir=data if data and not data.endswith(".zip") else None,
        epochs=epochs,
        batch_size=batch_size,
        use_colors=colors,
        base_model=base_model,
    )

    click.echo(f"Training complete! Best model: {weights}")


@train.command("fcnn")
@click.option(
    "--data",
    "-d",
    type=click.Path(exists=True),
    default="soccernet/tracking/tracking/train.zip",
    help="Path to SoccerNet tracking zip.",
)
@click.option(
    "--epochs",
    "-e",
    type=int,
    default=10,
    help="Number of training epochs.",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=64,
    help="Training batch size.",
)
@click.option(
    "--save-path",
    type=str,
    default="fcnn_player_tracker.pth",
    help="Path to save model weights.",
)
def train_fcnn_cmd(
    data: str,
    epochs: int,
    batch_size: int,
    save_path: str,
) -> None:
    """
    Train Faster R-CNN player detector.

    Faster R-CNN provides higher accuracy for small objects (distant players)
    compared to YOLO, with a latency trade-off.

    Example:
        $ torchkick train fcnn --epochs 10 --batch-size 64
    """
    from torchkick.training import train_fcnn

    click.echo("Training Faster R-CNN player detector")
    click.echo(f"  Data: {data}")
    click.echo(f"  Epochs: {epochs}")
    click.echo(f"  Batch size: {batch_size}")

    weights = train_fcnn(
        data_zip=data,
        epochs=epochs,
        batch_size=batch_size,
        save_path=save_path,
    )

    click.echo(f"Training complete! Model: {weights}")


@train.command("rtdetr")
@click.option(
    "--data",
    "-d",
    type=click.Path(exists=True),
    default="soccernet/tracking/tracking/train.zip",
    help="Path to SoccerNet tracking zip.",
)
@click.option(
    "--epochs",
    "-e",
    type=int,
    default=20,
    help="Number of training epochs.",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=32,
    help="Training batch size.",
)
@click.option(
    "--save-path",
    type=str,
    default="rtdetr_player_tracker.pth",
    help="Path to save model weights.",
)
def train_rtdetr_cmd(
    data: str,
    epochs: int,
    batch_size: int,
    save_path: str,
) -> None:
    """
    Train RT-DETR player detector.

    RT-DETR is a transformer-based detector with Apache 2.0 license,
    achieving YOLO-like speed with higher accuracy.

    Example:
        $ torchkick train rtdetr --epochs 20 --batch-size 32
    """
    from torchkick.training import train_rtdetr

    click.echo("Training RT-DETR player detector")
    click.echo(f"  Data: {data}")
    click.echo(f"  Epochs: {epochs}")
    click.echo(f"  Batch size: {batch_size}")

    weights = train_rtdetr(
        data_zip=data,
        epochs=epochs,
        batch_size=batch_size,
        save_path=save_path,
    )

    click.echo(f"Training complete! Model: {weights}")


@train.command("lines")
@click.option(
    "--data",
    "-d",
    type=click.Path(exists=True),
    required=True,
    help="Path to line detection training data.",
)
@click.option(
    "--output",
    "-o",
    type=str,
    default="weights/pitch",
    help="Output directory for weights.",
)
@click.option(
    "--epochs",
    "-e",
    type=int,
    default=100,
    help="Number of training epochs.",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=8,
    help="Training batch size.",
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    default="models/pitch/config/hrnetv2_w48_l.yaml",
    help="HRNet config file.",
)
def train_lines_cmd(
    data: str,
    output: str,
    epochs: int,
    batch_size: int,
    config: str,
) -> None:
    """
    Train pitch line detector.

    Trains HRNet-based line detection model for camera calibration.

    Example:
        $ torchkick train lines --data calibration_data/ --epochs 100
    """
    from torchkick.training import train_lines

    click.echo("Training pitch line detector")
    click.echo(f"  Data: {data}")
    click.echo(f"  Output: {output}")
    click.echo(f"  Epochs: {epochs}")

    weights = train_lines(
        data_dir=data,
        output_dir=output,
        config_path=config,
        epochs=epochs,
        batch_size=batch_size,
    )

    click.echo(f"Training complete! Best model: {weights}")


if __name__ == "__main__":
    main()
