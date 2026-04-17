"""
Task 1: Bundle Adjustment from scratch with PyTorch.

Inputs
------
- data/points2d.npz        : 50 keys "view_000"~"view_049", each (20000, 3) = (x, y, visibility)
- data/points3d_colors.npy : (20000, 3) RGB colors in [0, 1] (for OBJ export only)

Outputs
-------
- outputs/loss_curve.png     : reprojection loss vs. iteration
- outputs/reconstruction.obj : colored 3D point cloud ("v x y z r g b" per line)

Projection convention (from README)
-----------------------------------
    [Xc, Yc, Zc]^T = R @ [X, Y, Z]^T + T
    u = -f * Xc / Zc + cx
    v =  f * Yc / Zc + cy
The object is near the origin (Z ~= 0), and cameras sit on the +Z side looking
toward -Z, so T ~= [0, 0, -d] and Zc < 0 for visible points.
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# =========================================================================
# Config
# =========================================================================
DATA_DIR   = "data"
OUT_DIR    = "outputs"

IMG_SIZE   = 1024
N_VIEWS    = 50
N_POINTS   = 20000

# ---- optimization ----
N_ITERS    = 2000
# Parameter magnitudes differ a lot, so use separate learning rates per group.
LR_FOCAL   = 10.0    # focal is on the order of ~1000
LR_EULER   = 1e-2    # radians
LR_TRANS   = 1e-2
LR_POINTS  = 1e-2

# ---- initialization ----
INIT_FOV_DEG = 50.0  # rough FoV guess, used to derive the initial focal length
INIT_DEPTH   = 2.5   # T = [0, 0, -d]
INIT_PT_STD  = 0.1   # std of random init for 3D points

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================================
# Data loading
# =========================================================================
def load_observations(data_dir):
    """Load 2D observations from points2d.npz.

    Returns
    -------
    xy  : (V, N, 2) float tensor, observed pixel coordinates
    vis : (V, N)    float tensor, 1.0 if visible else 0.0
    """
    npz = np.load(os.path.join(data_dir, "points2d.npz"))
    arr = np.stack([npz[f"view_{i:03d}"] for i in range(N_VIEWS)], axis=0)  # (V, N, 3)
    xy  = torch.from_numpy(arr[..., :2]).float()
    vis = torch.from_numpy(arr[...,  2]).float()
    return xy, vis


# =========================================================================
# Rotation: Euler (XYZ) -> matrix
# =========================================================================
def euler_to_R(eulers):
    """Convert Euler angles (XYZ convention) to rotation matrices.

    eulers : (..., 3)   ->   R : (..., 3, 3),   R = Rx(a) @ Ry(b) @ Rz(c)
    """
    a, b, c = eulers.unbind(-1)
    zero = torch.zeros_like(a)
    one  = torch.ones_like(a)
    ca, sa = torch.cos(a), torch.sin(a)
    cb, sb = torch.cos(b), torch.sin(b)
    cc, sc = torch.cos(c), torch.sin(c)

    def stack3x3(r00, r01, r02, r10, r11, r12, r20, r21, r22):
        return torch.stack([
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ], dim=-2)

    Rx = stack3x3(one,  zero, zero,  zero,  ca,  -sa,  zero,  sa,   ca)
    Ry = stack3x3( cb,  zero,  sb,  zero,  one, zero,   -sb, zero,  cb)
    Rz = stack3x3( cc,   -sc, zero,   sc,   cc, zero,  zero, zero, one)
    return Rx @ Ry @ Rz


# =========================================================================
# Model
# =========================================================================
class BundleAdjustment(nn.Module):
    """Learnable parameters:
        focal    : scalar, shared across all cameras
        eulers   : (V, 3)  per-view rotation (XYZ Euler)
        trans    : (V, 3)  per-view translation
        points3d : (N, 3)  3D points in the world frame
    """

    def __init__(self, n_views, n_points, init_focal, init_depth,
                 init_pt_std=0.1, cx=None, cy=None):
        super().__init__()
        self.register_buffer("cx", torch.tensor(float(cx)))
        self.register_buffer("cy", torch.tensor(float(cy)))

        self.focal  = nn.Parameter(torch.tensor(float(init_focal)))
        self.eulers = nn.Parameter(torch.zeros(n_views, 3))

        T = torch.zeros(n_views, 3)
        T[:, 2] = -init_depth                       # cameras sit on +Z side and look toward -Z
        self.trans = nn.Parameter(T)

        self.points3d = nn.Parameter(torch.randn(n_points, 3) * init_pt_std)

    def project(self):
        """Project all 3D points into all views.

        Returns
        -------
        uv : (V, N, 2)  predicted pixel coordinates
        Zc : (V, N)     camera-frame depth (useful for sanity checks / masking)
        """
        R = euler_to_R(self.eulers)                                    # (V, 3, 3)
        # Xc = R @ X + T, broadcast over N points.
        Xc = torch.einsum("vij,nj->vni", R, self.points3d) \
             + self.trans.unsqueeze(1)                                 # (V, N, 3)
        X, Y, Z = Xc.unbind(-1)                                        # each (V, N)

        # Projection formula from README: u takes a minus sign, v does not.
        u = -self.focal * X / Z + self.cx
        v =  self.focal * Y / Z + self.cy
        return torch.stack([u, v], dim=-1), Z


# =========================================================================
# Loss
# =========================================================================
def reprojection_loss(pred_uv, obs_uv, vis_mask):
    """Mean squared reprojection error, masked by visibility.

    pred_uv, obs_uv : (V, N, 2)
    vis_mask        : (V, N)      1.0 if the observation is valid
    """
    diff2 = ((pred_uv - obs_uv) ** 2).sum(dim=-1)   # (V, N)  squared pixel distance
    n_valid = vis_mask.sum().clamp_min(1.0)
    return (diff2 * vis_mask).sum() / n_valid


# =========================================================================
# Training loop
# =========================================================================
def train(model, obs_uv, vis_mask, n_iters, log_every=100):
    optim = torch.optim.Adam([
        {"params": [model.focal],    "lr": LR_FOCAL},
        {"params": [model.eulers],   "lr": LR_EULER},
        {"params": [model.trans],    "lr": LR_TRANS},
        {"params": [model.points3d], "lr": LR_POINTS},
    ])

    losses = []
    for it in range(n_iters):
        optim.zero_grad()
        pred_uv, _ = model.project()
        loss = reprojection_loss(pred_uv, obs_uv, vis_mask)
        loss.backward()
        optim.step()

        losses.append(loss.item())
        if it % log_every == 0 or it == n_iters - 1:
            rmse = math.sqrt(loss.item())   # RMSE in pixel units
            print(f"iter {it:5d} | loss={loss.item():10.4f} | "
                  f"pixel RMSE={rmse:7.3f} | f={model.focal.item():7.2f}")

    return losses


# =========================================================================
# IO helpers
# =========================================================================
def plot_loss(losses, path):
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.yscale("log")
    plt.xlabel("iteration")
    plt.ylabel("reprojection loss (pixel^2)")
    plt.title("Bundle Adjustment - training loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def save_colored_obj(points, colors, path):
    """Write an OBJ with per-vertex color: "v x y z r g b" (r,g,b in [0,1])."""
    points = np.asarray(points, dtype=np.float32)
    colors = np.clip(np.asarray(colors, dtype=np.float32), 0.0, 1.0)
    with open(path, "w") as f:
        for (x, y, z), (r, g, b) in zip(points, colors):
            f.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.4f} {g:.4f} {b:.4f}\n")


# =========================================================================
# Main
# =========================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(0)

    # ---- data ----
    obs_uv, vis_mask = load_observations(DATA_DIR)
    obs_uv   = obs_uv.to(DEVICE)
    vis_mask = vis_mask.to(DEVICE)
    print(f"Loaded observations: {tuple(obs_uv.shape)}, "
          f"visible = {int(vis_mask.sum().item())}/{vis_mask.numel()}")

    # ---- model ----
    cx = cy = IMG_SIZE / 2
    init_focal = IMG_SIZE / (2 * math.tan(math.radians(INIT_FOV_DEG) / 2))
    print(f"Init focal (FoV={INIT_FOV_DEG} deg) = {init_focal:.2f}")

    model = BundleAdjustment(
        n_views=N_VIEWS, n_points=N_POINTS,
        init_focal=init_focal, init_depth=INIT_DEPTH,
        init_pt_std=INIT_PT_STD, cx=cx, cy=cy,
    ).to(DEVICE)

    # ---- optimize ----
    losses = train(model, obs_uv, vis_mask, n_iters=N_ITERS)

    # ---- save ----
    plot_loss(losses, os.path.join(OUT_DIR, "loss_curve.png"))

    pts    = model.points3d.detach().cpu().numpy()
    colors = np.load(os.path.join(DATA_DIR, "points3d_colors.npy"))
    save_colored_obj(pts, colors, os.path.join(OUT_DIR, "reconstruction.obj"))

    print(f"\nFinal focal  = {model.focal.item():.2f}")
    print(f"Saved: {OUT_DIR}/loss_curve.png, {OUT_DIR}/reconstruction.obj")


if __name__ == "__main__":
    main()