import pandas as pd
import matplotlib.pyplot as plt
import ast 
import numpy as np

import matplotlib.animation as animation


df = pd.read_csv("HumanBlood_28Frames_tifANDnpz/humanBlood_dataset.csv")

df["frame"] = df["frame"].astype(int)
frames = sorted(df["frame"].unique())

# for frame in frames:
#     df_frame = df[df["frame"] == frame]

#     plt.figure(figsize=(10, 10))
#     for i, row in df_frame.iterrows():
#         boundary = ast.literal_eval(row["boundary"])  
#         boundary = np.array(boundary)
#         cx, cy = row["x_center"], row["y_center"]

#         plt.plot(boundary[:, 1], boundary[:, 0], color='blue', alpha=0.4)
#         plt.plot(cx, cy, 'r.', markersize=2)

#     plt.title(f"Cell boundaries and centers-Frame {frame}")
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.gca().invert_yaxis()
#     plt.axis("equal")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()



#ANIMATION


fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, df["x_center"].max() + 10)
ax.set_ylim(df["y_center"].max() + 10, 0)  
ax.set_aspect("equal")
ax.set_title("Cell boundaries and centers (Animated)")
ax.axis("off")

boundary_lines = []
center_dots, = ax.plot([], [], 'r.', markersize=2)

def update(frame):
    ax.clear()
    df_frame = df[df["frame"] == frame]
    
    for _, row in df_frame.iterrows():
        boundary = np.array(ast.literal_eval(row["boundary"]))
        ax.plot(boundary[:, 1], boundary[:, 0], color='blue', alpha=0.4)

    ax.plot(df_frame["x_center"], df_frame["y_center"], 'r.', markersize=2)
    ax.set_title(f"Frame {frame}")
    ax.set_xlim(0, df["x_center"].max() + 10)
    ax.set_ylim(df["y_center"].max() + 10, 0)
    ax.set_aspect("equal")
    ax.axis("off")

ani = animation.FuncAnimation(fig, update, frames=frames, interval=500, repeat=True)

plt.show()