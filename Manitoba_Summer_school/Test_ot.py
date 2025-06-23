import numpy as np
import ot
import ot.plot
import matplotlib.pyplot as plt

# ---- Synthesize binary mask data (like in your real dataset) ----
mask_p = np.zeros((20, 20), dtype=np.int32)
mask_c = np.zeros((20, 20), dtype=np.int32)

# One square mask in parent
mask_p[5:10, 5:10] = 1  # one parent mask

# Two square masks in child
mask_c[4:7, 4:7] = 1    # child mask 1
mask_c[8:11, 8:11] = 2  # child mask 2

# ---- Convert masks to coordinates (argwhere) ----
coords_p = np.argwhere(mask_p > 0)  # shape (N_p, 2)
coords_c = np.argwhere(mask_c > 0)  # shape (N_c, 2)

# ---- Uniform weights for OT ----
a = np.ones(len(coords_p)) / len(coords_p)
b = np.ones(len(coords_c)) / len(coords_c)

# ---- Compute pairwise cost matrix ----
M = ot.dist(coords_p, coords_c, metric="sqeuclidean")
M = M.astype(np.float64)

M /= M.max()

# ---- Solve OT ----
T = ot.emd(a, b, M)

# ---- Plot: Coordinates ----
plt.figure(figsize=(7, 3))
plt.plot(coords_p[:, 1], coords_p[:, 0], "+b", label="Parent")
plt.plot(coords_c[:, 1], coords_c[:, 0], "xr", label="Child")
plt.gca().invert_yaxis()
plt.axis("equal")
plt.title("Synthetic Parent & Child Coordinates")
plt.legend()

# ---- Plot: Cost matrix ----
plt.figure(figsize=(5, 4))
plt.imshow(M, interpolation="nearest")
plt.title("Squared Euclidean Cost")
plt.colorbar()
plt.tight_layout()

# ---- Plot: OT plan arrows ----
plt.figure(figsize=(6, 5))
ot.plot.plot2D_samples_mat(coords_p, coords_c, T, c=[0.5, 0.5, 1])
plt.plot(coords_p[:, 1], coords_p[:, 0], "+b", label="Parent")
plt.plot(coords_c[:, 1], coords_c[:, 0], "xr", label="Child")
plt.gca().invert_yaxis()
plt.axis("equal")
plt.title("OT Plan: Parent → Child")
plt.legend()

plt.show()
