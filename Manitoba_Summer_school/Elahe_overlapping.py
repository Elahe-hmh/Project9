import numpy as np
import pandas as pd
from pathlib import Path
from skimage.measure import regionprops
from skimage.segmentation import relabel_sequential
import matplotlib.pyplot as plt

import matplotlib.cm as cm
from matplotlib import colormaps

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0

seg_dir = Path("HumanBlood_28Frames_tifANDnpz")
seg_files = sorted(seg_dir.glob("*_seg.npz"))
frame_keys = [int(f.stem.split("_")[0]) for f in seg_files]
frame_map = {int(f.stem.split("_")[0]): f for f in seg_files}

df_rows = []
track_id_counter = 0
prev_label_to_track = {}

for t in sorted(frame_keys):
    data = np.load(frame_map[t], allow_pickle=True)
    masks = relabel_sequential(data["masks"])[0]  
    props = regionprops(masks)
    
    label_to_track = {}
    for prop in props:
        label = prop.label
        centroid = prop.centroid
        area = prop.area

        matched = False

        if t > min(frame_keys):
            data_prev = np.load(frame_map[t-1], allow_pickle=True)
            masks_prev = relabel_sequential(data_prev["masks"])[0]
            for prev_label, prev_track_id in prev_label_to_track.items():
                mask1 = masks == label
                mask2 = masks_prev == prev_label
                iou = compute_iou(mask1, mask2)
                if iou > 0.4:
                    label_to_track[label] = prev_track_id
                    matched = True
                    break

        if not matched:
            label_to_track[label] = track_id_counter
            track_id_counter += 1

        df_rows.append({
            "frame": t,
            "cell_id": label,
            "x_center": centroid[1],
            "y_center": centroid[0],
            "area": area,
            "track_id": label_to_track[label]
        })

    prev_label_to_track = label_to_track

df = pd.DataFrame(df_rows)
df.to_csv("HumanBlood_28Frames_tifANDnpz/humanBlood_trackedOverlap.csv", index=False)
print(f"overlap tracking complete. number of tracks: {df['track_id'].nunique()}")



ddf = pd.read_csv("humanBlood_tracked_overlap.csv")
df = df[df["track_id"] != -1]

# Get unique track IDs and assign colormap
track_ids = df["track_id"].unique()
n_tracks = len(track_ids)
cmap = colormaps.get_cmap("turbo")
colors = [cmap(i / n_tracks) for i in range(n_tracks)]

plt.figure(figsize=(10, 10))

for i, track_id in enumerate(track_ids):
    track = df[df["track_id"] == track_id].sort_values("frame")
    color = colors[i]
    plt.plot(track["x_center"], track["y_center"], marker='o', color=color, alpha=0.7)

plt.title("Cell Tracking (Overlap-Based)")
plt.xlabel("X")
plt.ylabel("Y")
plt.gca().invert_yaxis()
plt.axis("equal")
plt.grid(True)
plt.tight_layout()
plt.show()


#ANIMATION


# import matplotlib.animation as animation

# # Load the overlap-tracked CSV
# df = pd.read_csv("HumanBlood_28Frames_tifANDnpz/humanBlood_trackedOverlap.csv")
# df = df[df["track_id"] != -1]  # only plot tracked cells

# # Get track IDs and assign colormap
# track_ids = df["track_id"].unique()
# n_tracks = len(track_ids)
# cmap = cm.get_cmap("turbo", n_tracks)
# track_colors = {track_id: cmap(i / n_tracks) for i, track_id in enumerate(track_ids)}

# # Set up figure
# fig, ax = plt.subplots(figsize=(8, 8))
# ax.set_xlim(0, df["x_center"].max() + 10)
# ax.set_ylim(df["y_center"].max() + 10, 0)  # invert y-axis
# ax.set_aspect("equal")
# ax.axis("off")

# # Animation update function
# def update(frame_num):
#     ax.clear()
#     ax.set_xlim(0, df["x_center"].max() + 10)
#     ax.set_ylim(df["y_center"].max() + 10, 0)
#     ax.set_aspect("equal")
#     ax.axis("off")
#     ax.set_title(f"Frame {frame_num}")

#     df_sub = df[df["frame"] <= frame_num]

#     for track_id in track_ids:
#         track = df_sub[df_sub["track_id"] == track_id].sort_values("frame")
#         color = track_colors[track_id]
#         ax.plot(track["x_center"], track["y_center"], marker='o', color=color, alpha=0.7)

# # Create animation
# ani = animation.FuncAnimation(fig, update, frames=sorted(df["frame"].unique()), interval=400)

# plt.show()
