import pandas as pd
from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("HumanBlood_28Frames_tifANDnpz/humanBlood_dataset.csv")
df["track_id"] = -1
df["frame"] = df["frame"].astype(int)

next_track_id = 0
max_dist = 15 #this is in pixels

frames = sorted(df["frame"].unique())

for i in range(len(frames) - 1):
    f_curr = frames[i]
    f_next = frames[i + 1]

    df_curr = df[df["frame"] == f_curr]
    df_next = df[df["frame"] == f_next]

    tree = cKDTree(df_next[["x_center", "y_center"]].values)

    for idx, row in df_curr.iterrows():
        x, y = row["x_center"], row["y_center"]
        dist, neighbor_idx = tree.query([x, y], distance_upper_bound=max_dist)

        if neighbor_idx < len(df_next):
            match_idx = df_next.index[neighbor_idx]

            if df.loc[idx, "track_id"] == -1:
                df.at[idx, "track_id"] = next_track_id
                df.at[match_idx, "track_id"] = next_track_id
                next_track_id += 1
            else:
                df.at[match_idx, "track_id"] = df.at[idx, "track_id"]


df_tracked = df[df["track_id"] != -1]

track_ids = df_tracked["track_id"].unique()
n_tracks = len(track_ids)
cmap = plt.get_cmap('viridis') #viridis\

n_tracks = df["track_id"].nunique()
print("Number of unique tracks:", n_tracks)

#save
df.to_csv("HumanBlood_28Frames_tifANDnpz/humanBlood_tracked_NN.csv", index=False)
print("tracking complete.")


plt.figure(figsize=(10, 10))

for i, track_id in enumerate(track_ids):
    track = df_tracked[df_tracked["track_id"] == track_id].sort_values("frame")
    color = cmap(i / n_tracks)
    plt.plot(track["x_center"], track["y_center"], marker='o', color=color, alpha=0.7)

plt.title("Cell tracking (colored by track ID)")
plt.xlabel("X")
plt.ylabel("Y")
plt.gca().invert_yaxis()
plt.axis("equal")
plt.grid(True)
plt.tight_layout()
plt.show()







#ANIMATION


import matplotlib.animation as animation

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, df["x_center"].max() + 10)
ax.set_ylim(df["y_center"].max() + 10, 0)
ax.set_aspect("equal")
ax.axis("off")

def update(frame_num):
    ax.clear()
    df_sub = df_tracked[df_tracked["frame"] <= frame_num]
    for i, track_id in enumerate(track_ids):
        track = df_sub[df_sub["track_id"] == track_id].sort_values("frame")
        color = cmap(i / n_tracks)
        ax.plot(track["x_center"], track["y_center"], marker='o', color=color, alpha=0.7)
    ax.set_title(f"Frame {frame_num}")
    ax.set_xlim(0, df["x_center"].max() + 10)
    ax.set_ylim(df["y_center"].max() + 10, 0)
    ax.axis("off")

ani = animation.FuncAnimation(fig, update, frames=sorted(df_tracked["frame"].unique()), interval=400)
plt.show()
