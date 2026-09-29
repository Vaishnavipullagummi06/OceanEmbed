"""
OceanEmbed Validation & Metrics Evaluation (SIH26066)
Evaluates trained OceanEmbed model on unseen test dataset against Climatology and SST-persistence.
Computes RMSE, Bias, and Correlation per depth, plus spatial maps and profile comparisons.
"""
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DEPTHS, ROI, SUB_REGIONS
from models.oceanembed import OceanEmbed
from data.synthetic import make_synthetic_dataset, generate_land_mask


def predict(model, ckpt, X, dev):
    """Normalized input inference -> inverse scaled predictions."""
    model.eval()
    xm, xs = ckpt["xm"], ckpt["xs"]
    ym, ys = ckpt["ym"], ckpt["ys"]
    mask = ckpt.get("mask", None)
    
    xn = (X - xm) / xs
    if mask is not None:
        xn = xn * mask[None, :, :]

    with torch.no_grad():
        p_norm = model(torch.tensor(xn, dtype=torch.float32).to(dev)).cpu().numpy()
        
    p = p_norm * ys + ym
    if mask is not None:
        p = p * mask[None, :, :]
    return p


def compute_metrics(pred, obs):
    """Computes RMSE, Bias, and Pearson Correlation between 1D arrays of predictions and observations."""
    diff = pred - obs
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    bias = float(np.mean(diff))
    
    std_p, std_o = np.std(pred), np.std(obs)
    if std_p > 1e-6 and std_o > 1e-6:
        corr = float(np.corrcoef(pred, obs)[0, 1])
    else:
        corr = 0.0
        
    return dict(RMSE=rmse, Bias=bias, Corr=corr)


def validate(model, ckpt, dev, n_test=60, pts=200):
    print("--- Starting OceanEmbed Independent Validation ---")
    cin = ckpt.get("cin", 11)
    include_coords = (cin > 7)
    
    X, Y, mask, dates = make_synthetic_dataset(
        n_samples=n_test,
        seed=999,
        include_coords=include_coords
    )

    P = predict(model, ckpt, X, dev)
    clim = ckpt["ym"]  # Climatology baseline (mean profile)

    ocean_points = np.where(mask[0, 0] > 0.5)
    n_ocean_pts = len(ocean_points[0])
    
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(n_ocean_pts, size=min(n_test * pts, n_ocean_pts * n_test), replace=True)
    
    b_idx = rng.integers(0, n_test, size=len(sample_indices))
    i_idx = ocean_points[0][sample_indices % n_ocean_pts]
    j_idx = ocean_points[1][sample_indices % n_ocean_pts]

    # Target observations with simulated ARGO sensor noise
    obs = Y[b_idx, :, i_idx, j_idx] + rng.normal(0, 0.05, (len(b_idx), len(DEPTHS)))
    
    pred_oceanembed = P[b_idx, :, i_idx, j_idx]
    pred_clim = np.broadcast_to(clim[0, :, 0, 0], obs.shape)
    pred_sst_pers = np.repeat(X[b_idx, 0, i_idx, j_idx][:, None], len(DEPTHS), axis=1)

    rows = []
    for k, d in enumerate(DEPTHS):
        for name, pred in [("OceanEmbed", pred_oceanembed), ("Climatology", pred_clim), ("SST-persistence", pred_sst_pers)]:
            metrics = compute_metrics(pred[:, k], obs[:, k])
            rows.append(dict(depth_m=d, model=name, **metrics))

    df = pd.DataFrame(rows)
    df.to_csv("metrics.csv", index=False)
    
    print("\n--- Key Depth Benchmark Results (°C) ---")
    summary_df = df[df.depth_m.isin([0, 50, 100, 200, 500, 1000])].round(3)
    print(summary_df.to_string(index=False))

    # Visualizations
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    
    # Plot 1: RMSE vs Depth
    for name in ["OceanEmbed", "Climatology", "SST-persistence"]:
        s = df[df.model == name]
        ax[0].plot(s.RMSE, s.depth_m, marker="o", linewidth=2, label=name)
    ax[0].invert_yaxis()
    ax[0].set_xlabel("RMSE (°C)", fontsize=11)
    ax[0].set_ylabel("Depth (m)", fontsize=11)
    ax[0].set_title("Validation RMSE vs Depth", fontsize=12, fontweight="bold")
    ax[0].grid(True, linestyle="--", alpha=0.6)
    ax[0].legend()

    # Plot 2 & 3: 100m Temperature Comparison Map
    k_100 = DEPTHS.index(100)
    vmin = min(Y[0, k_100][mask[0, 0] > 0.5].min(), P[0, k_100][mask[0, 0] > 0.5].min())
    vmax = max(Y[0, k_100][mask[0, 0] > 0.5].max(), P[0, k_100][mask[0, 0] > 0.5].max())

    im1 = ax[1].imshow(np.where(mask[0, 0] > 0.5, Y[0, k_100], np.nan), cmap="turbo", vmin=vmin, vmax=vmax)
    ax[1].set_title("Ground Truth Temp @ 100m (°C)", fontsize=12)
    plt.colorbar(im1, ax=ax[1], shrink=0.8)

    im2 = ax[2].imshow(np.where(mask[0, 0] > 0.5, P[0, k_100], np.nan), cmap="turbo", vmin=vmin, vmax=vmax)
    ax[2].set_title("OceanEmbed Temp @ 100m (°C)", fontsize=12)
    plt.colorbar(im2, ax=ax[2], shrink=0.8)

    plt.tight_layout()
    plt.savefig("results.png", dpi=130)
    print("Saved metrics.csv and results.png successfully.")


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        ckpt = torch.load("oceanembed.pt", weights_only=False, map_location=dev)
        cin = ckpt.get("cin", 11)
        model = OceanEmbed(cin=cin).to(dev)
        model.load_state_dict(ckpt["model"])
        validate(model, ckpt, dev)
    except FileNotFoundError:
        print("oceanembed.pt not found. Run train.py first!")
