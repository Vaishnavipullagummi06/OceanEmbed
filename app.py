"""
OceanEmbed Streamlit Web Application (SIH26066)
Interactive visualization of subsurface ocean temperature predictions at 15 depths over the North Indian Ocean.
"""
import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt

from config import DEPTHS, VARS, SUB_REGIONS, ROI
from models.oceanembed import OceanEmbed
from data.synthetic import make_synthetic_dataset, generate_land_mask
from data.argo import fetch_argo_profiles
from validate import predict

st.set_page_config(
    page_title="OceanEmbed | Subsurface Ocean Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for premium look
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E88E5; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.1rem; color: #546E7A; margin-bottom: 1.5rem; }
    .metric-card { background-color: #F0F4F8; border-radius: 8px; padding: 15px; border-left: 5px solid #1E88E5; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🌊 OceanEmbed – Subsurface Ocean Temperature Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">SIH26066 | Predicting 15 subsurface depth temperatures from satellite surface fields over the North Indian Ocean</div>', unsafe_allow_html=True)


@st.cache_resource
def load_model_checkpoint():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        ck = torch.load("oceanembed.pt", map_location=dev, weights_only=False)
        cin = ck.get("cin", 11)
        model = OceanEmbed(cin=cin).to(dev)
        model.load_state_dict(ck["model"])
        return model, ck, dev
    except FileNotFoundError:
        st.warning("`oceanembed.pt` checkpoint not found. Using randomly initialized demo model.")
        model = OceanEmbed(cin=11).to(dev)
        mask = generate_land_mask()
        ck = {
            "model": model.state_dict(),
            "xm": np.zeros((1, 11, 1, 1), dtype=np.float32),
            "xs": np.ones((1, 11, 1, 1), dtype=np.float32),
            "ym": np.full((1, 15, 1, 1), 15.0, dtype=np.float32),
            "ys": np.ones((1, 15, 1, 1), dtype=np.float32),
            "mask": mask,
            "cin": 11,
        }
        return model, ck, dev


model, ckpt, dev = load_model_checkpoint()

# Sidebar Controls
st.sidebar.header("⚙️ Simulation Controls")

region = st.sidebar.selectbox("Region Selector", list(SUB_REGIONS.keys()), index=0)
scene_seed = st.sidebar.number_input("Day / Scene Index", min_value=0, max_value=365, value=180, step=1)
depth = st.sidebar.select_slider("Target Depth (m)", options=DEPTHS, value=100)

# Fetch Data
sub_bounds = SUB_REGIONS[region]
cin = ckpt.get("cin", 11)
include_coords = (cin > 7)

X, Y, mask, dates = make_synthetic_dataset(n_samples=1, seed=int(scene_seed) + 100, include_coords=include_coords)
P = predict(model, ckpt, X, dev)

# Crop to Selected Sub-Region
lats = np.linspace(ROI["lat_min"], ROI["lat_max"], X.shape[2])
lons = np.linspace(ROI["lon_min"], ROI["lon_max"], X.shape[3])

lat_idx = np.where((lats >= sub_bounds["lat_min"]) & (lats <= sub_bounds["lat_max"]))[0]
lon_idx = np.where((lons >= sub_bounds["lon_min"]) & (lons <= sub_bounds["lon_max"]))[0]

X_sub = X[:, :, lat_idx[:, None], lon_idx]
Y_sub = Y[:, :, lat_idx[:, None], lon_idx]
P_sub = P[:, :, lat_idx[:, None], lon_idx]
mask_sub = mask[:, :, lat_idx[:, None], lon_idx]

sub_lats = lats[lat_idx]
sub_lons = lons[lon_idx]

k = DEPTHS.index(depth)

# Tab Navigation
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ 2D Spatial Fields & ARGO", "📈 Vertical Profile Probe", "📊 Benchmark Metrics", "ℹ️ Architecture & Specs"])

with tab1:
    st.subheader(f"Spatial Temperature Field @ {depth}m Depth ({region})")
    
    c1, c2, c3 = st.columns(3)
    
    # Surface SST Input
    with c1:
        fig, ax = plt.subplots(figsize=(5, 4))
        sst_img = np.where(mask_sub[0, 0] > 0.5, X_sub[0, 0], np.nan)
        im = ax.imshow(sst_img, cmap="turbo", extent=[sub_lons[0], sub_lons[-1], sub_lats[0], sub_lats[-1]], origin="lower")
        ax.set_title("Surface Input SST (°C)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        plt.colorbar(im, ax=ax, shrink=0.8)
        st.pyplot(fig)

    # Ground Truth GLORYS
    vmin, vmax = float(np.nanmin(Y_sub[0, k])), float(np.nanmax(Y_sub[0, k]))
    with c2:
        fig, ax = plt.subplots(figsize=(5, 4))
        truth_img = np.where(mask_sub[0, 0] > 0.5, Y_sub[0, k], np.nan)
        im = ax.imshow(truth_img, cmap="turbo", extent=[sub_lons[0], sub_lons[-1], sub_lats[0], sub_lats[-1]], origin="lower", vmin=vmin, vmax=vmax)
        
        # Overlay ARGO Profiles
        argo_list = fetch_argo_profiles(region=region)
        argo_lons = [p["longitude"] for p in argo_list]
        argo_lats = [p["latitude"] for p in argo_list]
        ax.scatter(argo_lons, argo_lats, c="white", edgecolors="black", s=25, marker="o", label="ARGO Floats")
        
        ax.set_title(f"Truth (GLORYS) @ {depth}m", fontsize=11, fontweight="bold")
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.legend(loc="lower right", fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.8)
        st.pyplot(fig)

    # OceanEmbed Prediction
    with c3:
        fig, ax = plt.subplots(figsize=(5, 4))
        pred_img = np.where(mask_sub[0, 0] > 0.5, P_sub[0, k], np.nan)
        im = ax.imshow(pred_img, cmap="turbo", extent=[sub_lons[0], sub_lons[-1], sub_lats[0], sub_lats[-1]], origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(f"OceanEmbed Pred @ {depth}m", fontsize=11, fontweight="bold")
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        plt.colorbar(im, ax=ax, shrink=0.8)
        st.pyplot(fig)

    # Per-scene error metric card
    valid_err = np.abs(P_sub[0, k] - Y_sub[0, k])[mask_sub[0, 0] > 0.5]
    scene_rmse = np.sqrt(np.mean(valid_err ** 2)) if len(valid_err) > 0 else 0.0
    st.info(f"**Scene Evaluation @ {depth}m**: RMSE = `{scene_rmse:.3f} °C` over {region} ocean grid points.")

with tab2:
    st.subheader("Interactive Vertical Temperature Profile Probe")
    
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        probe_lat = st.slider("Select Latitude (°N)", float(sub_lats[0]), float(sub_lats[-1]), float(np.median(sub_lats)), step=0.25)
    with col_sel2:
        probe_lon = st.slider("Select Longitude (°E)", float(sub_lons[0]), float(sub_lons[-1]), float(np.median(sub_lons)), step=0.25)

    r_i = int(np.argmin(np.abs(sub_lats - probe_lat)))
    r_j = int(np.argmin(np.abs(sub_lons - probe_lon)))

    is_ocean = (mask_sub[0, 0, r_i, r_j] > 0.5)

    if not is_ocean:
        st.warning(f"Selected point ({probe_lat:.2f}°N, {probe_lon:.2f}°E) is over LAND.")
    else:
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Ground Truth profile
        truth_profile = Y_sub[0, :, r_i, r_j]
        pred_profile = P_sub[0, :, r_i, r_j]
        
        ax.plot(truth_profile, DEPTHS, "o-", color="#1976D2", linewidth=2.5, label="Ground Truth (GLORYS)")
        ax.plot(pred_profile, DEPTHS, "s--", color="#E53935", linewidth=2.5, label="OceanEmbed Prediction")
        
        # Climatology Baseline profile
        clim_profile = ckpt["ym"][0, :, 0, 0]
        ax.plot(clim_profile, DEPTHS, ":", color="#757575", linewidth=2, label="Climatology Baseline")

        ax.invert_yaxis()
        ax.set_xlabel("Temperature (°C)", fontsize=11)
        ax.set_ylabel("Depth (m)", fontsize=11)
        ax.set_title(f"Subsurface Profile at {sub_lats[r_i]:.2f}°N, {sub_lons[r_j]:.2f}°E", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        
        st.pyplot(fig)

with tab3:
    st.subheader("Model Benchmark Metrics (vs ARGO / GLORYS Independent Test)")
    
    try:
        metrics_df = pd.read_csv("metrics.csv")
        
        # Format table
        st.dataframe(metrics_df.style.highlight_min(axis=0, subset=["RMSE"], color="#C8E6C9"), use_container_width=True)
        
        # Metric depth plot
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for model_name in metrics_df["model"].unique():
            sub_df = metrics_df[metrics_df["model"] == model_name]
            ax.plot(sub_df["RMSE"], sub_df["depth_m"], marker="o", label=model_name)
            
        ax.invert_yaxis()
        ax.set_xlabel("RMSE (°C)")
        ax.set_ylabel("Depth (m)")
        ax.set_title("RMSE Comparison across all 15 Target Depths")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        st.pyplot(fig)
        
    except FileNotFoundError:
        st.info("Run `python validate.py` to generate the latest metrics summary.")

with tab4:
    st.subheader("Architecture & Data Pipeline Specifications")
    st.markdown("""
    - **Encoder**: U-Net architecture ("Ocean Embedding") extracting 64-dim latent representations from 11 surface features.
    - **Surface Inputs**: SST, SSS, SLA/SSH, Surface U/V Currents, U/V Winds, Normalized Lat/Lon, Sin/Cos DOY.
    - **Target Depths**: 0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 750, 1000 meters.
    - **Resolution & Grid**: Daily 0.25° grid over North Indian Ocean (5–25°N, 45–100°E).
    - **Validation**: Evaluated against gridded ARGO float observations and GLORYS12V1 reanalysis.
    """)