"""
OceanEmbed Training Script (SIH26066)
Trains U-Net encoder-decoder model on ocean surface inputs to predict 15 subsurface temperatures.
Supports synthetic & real Copernicus data, land-masked loss, and temporal train/test split.

Usage:
    python train.py --epochs 15 --mode synthetic
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from config import DEFAULT_HP, FEATURE_CHANNELS, VARS, DEPTHS
from models.oceanembed import OceanEmbed, masked_mse_loss
from data.synthetic import make_synthetic_dataset


def train(args, dev):
    print(f"--- Starting OceanEmbed Training [Mode: {args.mode.upper()}] ---")
    
    if args.mode == "real":
        try:
            from data.real import load_real_dataset
            print("Loading real dataset from Copernicus Marine / cached netCDF...")
            X, Y, mask, dates = load_real_dataset(year=args.year)
        except Exception as e:
            print(f"Warning: Failed to load real data ({e}). Falling back to synthetic dataset.")
            args.mode = "synthetic"

    if args.mode == "synthetic":
        include_coords = (args.cin > len(VARS))
        X, Y, mask, dates = make_synthetic_dataset(
            n_samples=args.n_train,
            seed=0,
            include_coords=include_coords
        )

    N, C, H, W = X.shape
    print(f"Dataset loaded: X={X.shape}, Y={Y.shape}, Mask={mask.shape}")

    # Compute land-masked normalization statistics
    valid_pixel_mask = (mask > 0.5)
    
    # Broadcast mask for statistics computation
    xm = np.zeros((1, C, 1, 1), dtype=np.float32)
    xs = np.ones((1, C, 1, 1), dtype=np.float32)
    for c in range(C):
        vals = X[:, c:c+1][valid_pixel_mask]
        if len(vals) > 0:
            xm[0, c, 0, 0] = vals.mean()
            xs[0, c, 0, 0] = vals.std() + 1e-6

    ym = np.zeros((1, len(DEPTHS), 1, 1), dtype=np.float32)
    ys = np.ones((1, len(DEPTHS), 1, 1), dtype=np.float32)
    for d in range(len(DEPTHS)):
        vals = Y[:, d:d+1][valid_pixel_mask]
        if len(vals) > 0:
            ym[0, d, 0, 0] = vals.mean()
            ys[0, d, 0, 0] = vals.std() + 1e-6

    Xn = (X - xm) / xs * mask
    Yn = (Y - ym) / ys * mask

    # TEMPORAL SPLIT: train on earlier timestamps (80%), validate on later timestamps (20%)
    split_idx = int(0.8 * N)
    Xt, Yt, Mt = Xn[:split_idx], Yn[:split_idx], mask[:split_idx]
    Xv, Yv, Mv = Xn[split_idx:], Yn[split_idx:], mask[split_idx:]
    print(f"Temporal split: {len(Xt)} train samples, {len(Xv)} validation samples (latest dates)")

    model = OceanEmbed(cin=C, emb=args.embed_dim, nd=len(DEPTHS)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    Xt_t, Yt_t, Mt_t = torch.tensor(Xt), torch.tensor(Yt), torch.tensor(Mt)
    Xv_t, Yv_t, Mv_t = torch.tensor(Xv).to(dev), torch.tensor(Yv).to(dev), torch.tensor(Mv).to(dev)

    best_val_loss = float("inf")
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt_t))
        tot_loss = 0.0
        
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb, yb, mb = Xt_t[idx].to(dev), Yt_t[idx].to(dev), Mt_t[idx].to(dev)
            
            pred = model(xb)
            loss = masked_mse_loss(pred, yb, mb)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            tot_loss += loss.item() * len(idx)

        sched.step()
        
        model.eval()
        with torch.no_grad():
            val_pred = model(Xv_t)
            val_loss = masked_mse_loss(val_pred, Yv_t, Mv_t).item()

        train_loss = tot_loss / len(perm)
        print(f"Epoch {ep:02d}/{args.epochs:02d} | Train MSE: {train_loss:.5f} | Val MSE (Temporal): {val_loss:.5f}")

    ckpt = {
        "model": model.state_dict(),
        "xm": xm,
        "xs": xs,
        "ym": ym,
        "ys": ys,
        "mask": mask[0],
        "cin": C,
        "mode": args.mode,
    }
    torch.save(ckpt, "oceanembed.pt")
    print("Training complete! Model saved to oceanembed.pt")
    return model, ckpt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OceanEmbed Model Trainer")
    parser.add_argument("--epochs", type=int, default=DEFAULT_HP["epochs"])
    parser.add_argument("--n_train", type=int, default=DEFAULT_HP["n_train"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_HP["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_HP["learning_rate"])
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_HP["weight_decay"])
    parser.add_argument("--embed_dim", type=int, default=DEFAULT_HP["embed_dim"])
    parser.add_argument("--cin", type=int, default=11, help="Number of input channels (7 surface or 11 with coords/doy)")
    parser.add_argument("--mode", type=str, choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--year", type=int, default=2022, help="Year to train real data on")
    
    args = parser.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train(args, dev)
