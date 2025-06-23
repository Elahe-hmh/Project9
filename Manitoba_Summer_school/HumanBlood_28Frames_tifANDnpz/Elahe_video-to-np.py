import os
import skvideo.io
import imageio
import numpy as np
np.float = float
np.int = int


video_file = "human_blood.avi"
output_dir = "frames_human_blood"

os.makedirs(output_dir, exist_ok=True)

video = skvideo.io.vread(video_file)

print(f"[INFO] Extracted {video.shape[0]} frames from {video_file}")

for i, frame in enumerate(video):
    grayscale = frame[:, :, 2] // 2 + frame[:, :, 0] // 2
    filename = os.path.join(output_dir, f"{i:04d}.tif")
    imageio.imwrite(filename, grayscale)
