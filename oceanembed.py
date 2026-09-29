"""
OceanEmbed Main Entrypoint (SIH26066)
Provides backwards-compatible entry point for oceanembed.py while using modular package components.
"""
import argparse
import torch
from config import DEPTHS, VARS
from models.oceanembed import OceanEmbed
from data.synthetic import make_synthetic_dataset as make_dataset
from train import train
from validate import validate, predict

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--n_train", type=int, default=400)
    args = ap.parse_args()
    args.mode = "synthetic"
    args.batch_size = 16
    args.lr = 2e-3
    args.weight_decay = 1e-4
    args.embed_dim = 64
    args.cin = 11

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = train(args, dev)
    validate(model, ckpt, dev)
