import argparse
import os
from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description="Upload ONNX models to Hugging Face Hub")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo ID (e.g., your-username/torchkick-models)")
    parser.add_argument("--token", default=None, help="HF Token (optional if already logged in via huggingface-cli)")
    args = parser.parse_args()

    api = HfApi(token=args.token)

    print(f"Ensuring repository '{args.repo_id}' exists...")
    api.create_repo(repo_id=args.repo_id, exist_ok=True, repo_type="model")

    # Define the models we want to upload
    models_to_upload = [
        {
            "local_path": "models/players/finetune/run/weights/best_fp16.onnx",
            "repo_path": "yolo11l_football_players_fp16.onnx",
        },
        {"local_path": "models/keypoints/finetune/last_fp16.onnx", "repo_path": "dinov2_pitch_keypoints_fp16.onnx"},
    ]

    for model in models_to_upload:
        local_path = model["local_path"]
        repo_path = model["repo_path"]

        if not os.path.exists(local_path):
            print(f"[Warning] Could not find {local_path}. Skipping.")
            continue

        print(f"Uploading {local_path} -> {repo_path}...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=args.repo_id,
        )
        print(f"Successfully uploaded {repo_path}!")

    print(f"\nAll models successfully uploaded to: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
