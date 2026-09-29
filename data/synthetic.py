"""
Synthetic Dataset Generator for OceanEmbed (SIH26066)
Generates physical synthetic ocean surface fields and subsurface temperature profiles
with realistic thermocline dynamics, land masking, and coordinate channels.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from config import DEPTHS, ROI, RAW_H, RAW_W


def smooth(rng, shape, sigma):
    f = gaussian_filter(rng.standard_normal(shape), sigma, mode="wrap")
    return f / (f.std() + 1e-8)


def generate_land_mask(h=RAW_H, w=RAW_W):
    """
    Generate realistic land mask for North Indian Ocean (5-25°N, 45-100°E).
    1 = ocean, 0 = land (Indian Peninsula, Arabian Peninsula, SE Asia).
    """
    lats = np.linspace(ROI["lat_min"], ROI["lat_max"], h)
    lons = np.linspace(ROI["lon_min"], ROI["lon_max"], w)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    ocean_mask = np.ones((h, w), dtype=np.float32)

    # Indian peninsula land polygon approximation (~8-23°N, ~72-85°E wedge)
    india_wedge = (lat_grid >= 8.0) & (lat_grid <= 23.0) & \
                  (lon_grid >= 68.0 + (lat_grid - 8.0) * 0.5) & \
                  (lon_grid <= 88.0 - (lat_grid - 8.0) * 0.7)
    ocean_mask[india_wedge] = 0.0

    # Arabian Peninsula (West of ~58°E, North of ~15°N)
    arabia = (lon_grid <= 58.0) & (lat_grid >= 15.0 - (58.0 - lon_grid) * 0.3)
    ocean_mask[arabia] = 0.0

    # SE Asia land (East of ~95°E, North of ~10°N)
    se_asia = (lon_grid >= 95.0) & (lat_grid >= 10.0 + (lon_grid - 95.0) * 0.8)
    ocean_mask[se_asia] = 0.0

    return ocean_mask


def make_synthetic_sample(rng, day_of_year=180, include_coords=True, h=RAW_H, w=RAW_W):
    """
    Generate a single synthetic day observation:
    Surface inputs X: (C, H, W)
    Target temperature Y: (15, H, W)
    Land Mask: (1, H, W)
    """
    lats = np.linspace(ROI["lat_min"], ROI["lat_max"], h)
    lons = np.linspace(ROI["lon_min"], ROI["lon_max"], w)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    mask = generate_land_mask(h, w)

    # Physical Surface Input Signals
    sla = 0.15 * smooth(rng, (h, w), 6)                                    # Sea Level Anomaly (m)
    sst = 29.5 - 0.12 * (lat_grid - 5) + 0.5 * smooth(rng, (h, w), 8)        # SST (°C)
    # Seasonal cycle on SST
    doy_rad = 2 * np.pi * day_of_year / 365.25
    sst += 1.5 * np.sin(doy_rad - 0.5)

    sss = 35.0 - 1.2 * smooth(rng, (h, w), 10) + 0.02 * (lat_grid - 5)       # SSS (PSU)
    u = -np.gradient(sla, axis=0) * 8                                      # Surface U current (m/s)
    v = np.gradient(sla, axis=1) * 8                                       # Surface V current (m/s)
    
    # Wind fields (SW monsoon in summer, NE monsoon in winter)
    monsoon_sign = np.cos(doy_rad)
    wu = 3 + 4 * smooth(rng, (h, w), 12) + 2.0 * monsoon_sign
    wv = 4 * smooth(rng, (h, w), 12) + 3.0 * monsoon_sign

    # Hidden Thermocline Physics
    zt = np.clip(90 + 400 * sla - 3 * np.hypot(wu, wv), 20, 250)
    z = np.array(DEPTHS, dtype=float)[:, None, None]
    
    tdeep = 5 + 8 * np.exp(-z / 400)
    f = 0.5 * (1 - np.tanh((z - zt) / 60))
    f0 = 0.5 * (1 - np.tanh((0 - zt) / 60))
    f = np.minimum(1.0, f / f0)
    
    T = tdeep + (sst - tdeep) * f
    T += 0.15 * gaussian_filter(rng.standard_normal(T.shape), (0, 2, 2))
    
    # Apply land mask to target
    T = T * mask[None, :, :]

    # Stack Input Channels
    feature_list = [sst, sss, sla, u, v, wu, wv]
    if include_coords:
        # Normalized lat/lon grids [-1, 1]
        norm_lat = (lat_grid - ROI["lat_min"]) / (ROI["lat_max"] - ROI["lat_min"]) * 2 - 1
        norm_lon = (lon_grid - ROI["lon_min"]) / (ROI["lon_max"] - ROI["lon_min"]) * 2 - 1
        sin_doy = np.full((h, w), np.sin(doy_rad))
        cos_doy = np.full((h, w), np.cos(doy_rad))
        feature_list.extend([norm_lat, norm_lon, sin_doy, cos_doy])

    X = np.stack(feature_list).astype(np.float32)
    # Apply land mask to inputs
    X = X * mask[None, :, :]
    
    return X, T.astype(np.float32), mask.astype(np.float32)


def make_synthetic_dataset(n_samples=400, seed=0, include_coords=True, h=RAW_H, w=RAW_W):
    """
    Generates a full synthetic dataset of n_samples.
    Returns: X (N, C, H, W), Y (N, 15, H, W), Masks (N, 1, H, W), Dates/DOYs (N,)
    """
    rng = np.random.default_rng(seed)
    xs, ys, masks = [], [], []
    doys = []
    
    for i in range(n_samples):
        doy = (i % 365) + 1
        x, y, m = make_synthetic_sample(rng, day_of_year=doy, include_coords=include_coords, h=h, w=w)
        xs.append(x)
        ys.append(y)
        masks.append(m[None, :, :])
        doys.append(doy)

    return (
        np.stack(xs),
        np.stack(ys),
        np.stack(masks),
        np.array(doys),
    )
