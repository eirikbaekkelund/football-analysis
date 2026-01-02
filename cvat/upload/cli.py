import argparse
import json
import os
from pathlib import Path
from glob import glob
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Generate CVAT pre-labeled annotations from football video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Save annotations locally (JSON + MOT format)
  python -m cvat.upload.cli match.mp4 -o output/

  # Upload to CVAT task using API token
  python -m cvat.upload.cli match.mp4 --task-id 123 --token YOUR_API_TOKEN

  # Process first 60 seconds only
  python -m cvat.upload.cli match.mp4 --task-id 123 --token TOKEN --duration 60

  # Use credentials file for CVAT authentication
  python -m cvat.upload.cli match.mp4 --task-id 123 --credentials ~/.cvat_creds.json

Credentials file format (JSON):
  {"host": "https://app.cvat.ai", "token": "YOUR_TOKEN"}
        """,
    )

    parser.add_argument(
        "videos",
        nargs="+",
        help="Video file(s) to process (supports glob patterns)",
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=["yolo", "fcnn"],
        default="fcnn",
        help="Detection model to use (default: fcnn)",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=None,
        help="Maximum duration in seconds (default: process full video)",
    )

    # CVAT connection options
    cvat_group = parser.add_argument_group("CVAT Upload Options")
    cvat_group.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="CVAT task ID to upload annotations to",
    )
    cvat_group.add_argument(
        "--create-task",
        action="store_true",
        help="Create a new CVAT task and upload video before annotating",
    )
    cvat_group.add_argument(
        "--project-id",
        type=int,
        default=None,
        help="CVAT project ID for new task (use with --create-task)",
    )
    cvat_group.add_argument(
        "--host",
        default="https://app.cvat.ai",
        help="CVAT host URL (default: https://app.cvat.ai)",
    )
    cvat_group.add_argument(
        "--token",
        default=None,
        help="CVAT API token (or use --credentials)",
    )
    cvat_group.add_argument(
        "--credentials",
        type=str,
        default=None,
        help="Path to JSON credentials file with 'host' and 'token' fields",
    )
    cvat_group.add_argument(
        "--use-shapes",
        action="store_true",
        help="Upload as individual shapes instead of tracks (not recommended)",
    )

    args = parser.parse_args()

    # Load credentials from file if provided
    host = args.host
    token = args.token

    if args.credentials:
        creds_path = Path(args.credentials).expanduser()
        if creds_path.exists():
            with open(creds_path) as f:
                creds = json.load(f)
                host = creds.get("host", host)
                token = creds.get("token", token)
                print(f"[Config] Loaded credentials from {creds_path}")
        else:
            print(f"[Warning] Credentials file not found: {creds_path}")

    # Also check environment variables
    if not token:
        token = os.environ.get("CVAT_TOKEN", "")
    if host == "https://app.cvat.ai":
        host = os.environ.get("CVAT_HOST", host)

    # Expand glob patterns
    video_paths = []
    for pattern in args.videos:
        expanded = glob(pattern)
        if expanded:
            video_paths.extend(expanded)
        else:
            video_paths.append(pattern)

    # Validate videos exist
    valid_paths = []
    for vp in video_paths:
        if Path(vp).exists():
            valid_paths.append(vp)
        else:
            print(f"[Warning] Video not found: {vp}")

    if not valid_paths:
        print("[Error] No valid video files found")
        return 1

    # Check CVAT upload requirements
    if args.task_id and not token:
        print("[Error] --token or --credentials required when using --task-id")
        return 1

    # Import here to avoid slow startup for --help
    from cvat.upload.prelabel_pipeline import (
        process_video_for_cvat,
        upload_tracks_with_client,
        upload_with_client,
    )

    # Create CVAT client if uploading
    client = None
    needs_cvat = args.task_id or args.create_task
    if needs_cvat and token:
        from cvat.cvat_integration import Client, Credentials

        creds = Credentials(
            host=host,
            username="",  # Not used with token auth
            password=token,
            use_token=True,
        )
        try:
            client = Client(creds)
        except Exception as e:
            print(f"[Error] Failed to connect to CVAT: {e}")
            return 1

    # Process each video
    all_results = {}

    for video_path in valid_paths:
        video_name = Path(video_path).stem
        print(f"\n{'='*60}")
        print(f"Processing: {video_name}")
        print("=" * 60)

        try:
            # Run the 3-pass pipeline
            annotations, metadata = process_video_for_cvat(
                video_path,
                max_duration=args.duration,
                model_type=args.model,
            )

            # Determine task ID - create new task if requested
            task_id = args.task_id

            if client and args.create_task:
                print(f"\n[CVAT] Creating new task for {video_name}...")
                from cvat.cvat_integration import FOOTBALL_LABELS

                task = client.create_task(
                    name=f"Pre-labeled: {video_name}",
                    project_id=args.project_id,
                    labels=FOOTBALL_LABELS if not args.project_id else None,
                )
                task_id = task.id
                print(f"[CVAT] Created task {task_id}")

                # Calculate stop frame based on duration
                stop_frame = None
                if args.duration:
                    stop_frame = int(metadata["fps"] * args.duration)
                    print(f"[CVAT] Uploading first {stop_frame} frames ({args.duration}s) of {video_path}...")
                else:
                    print(f"[CVAT] Uploading video {video_path}...")

                client.upload_video_to_task(
                    task_id,
                    video_path,
                    stop_frame=stop_frame,
                )
                print("[CVAT] Video uploaded")

            # Upload annotations to CVAT if configured
            if client and task_id:
                print(f"\n[Upload] Uploading annotations to CVAT task {task_id}...")
                try:
                    if args.use_shapes:
                        upload_with_client(client, task_id, annotations)
                    else:
                        upload_tracks_with_client(client, task_id, annotations)
                    print(f"[Upload] View at: {host}/tasks/{task_id}")
                except Exception as e:
                    print(f"[Error] Upload failed: {e}")
                    import traceback

                    traceback.print_exc()

            all_results[video_name] = annotations

            # Summary
            n_players = len([a for a in annotations if a.label == "player"])
            n_goalies = len([a for a in annotations if a.label == "goalkeeper"])
            n_refs = len([a for a in annotations if a.label == "referee"])

            print(f"\nSummary for {video_name}:")
            print(f"  Players: {n_players}")
            print(f"  Goalkeepers: {n_goalies}")
            print(f"  Referees: {n_refs}")
            print(f"  Total slots: {n_players + n_goalies + n_refs}")

        except Exception as e:
            print(f"[Error] Failed to process {video_path}: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Processed {len(all_results)} videos")
    if client and args.task_id:
        print(f"Uploaded to CVAT task: {args.task_id}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    main()
