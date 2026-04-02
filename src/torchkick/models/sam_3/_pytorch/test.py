import torch
from sam3 import Sam3VideoPredictor

# 1. Setup the predictor
device = "cuda" if torch.cuda.is_available() else "cpu"
predictor = Sam3VideoPredictor.from_pretrained("facebook/sam3").to(device)

# Path to your video (can be a path to a video file or directory of frames)
video_path = "/path/to/soccer_game.mp4"

# Define your team prompts based on jersey colors/kits
team_prompts = ["soccer player in white jersey", "soccer player in red jersey"]  # Team A  # Team B

all_team_masks = {}

# 2. Loop through each team separately
for team_prompt in team_prompts:
    print(f"Tracking: {team_prompt}")

    # Initialize a fresh session for this concept
    with predictor.inference_session(video_path) as session:
        # Add the text prompt to the first frame (frame_idx=0)
        # SAM3 will find ALL objects matching this text in the frame
        predictor.add_text_prompt(session=session, text=team_prompt, frame_index=0)

        # Propagate the masks through the entire video
        # This returns a dictionary of masks for every frame
        video_segments = predictor.propagate_in_video(session)

        # Store results
        all_team_masks[team_prompt] = video_segments

# 3. Post-Processing (Conceptual)
# You now have 'all_team_masks["...white..."]' and 'all_team_masks["...red..."]'
# You can iterate through frames and overlay these masks with different colors.
