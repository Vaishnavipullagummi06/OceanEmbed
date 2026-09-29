"""
OceanEmbed Configuration (SIH26066)
Centralized configuration parameters for depths, region, grid, and model hyperparameters.
"""

# Standard 15 target depths (in meters)
DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 750, 1000]

# Feature variables
VARS = ["SST", "SSS", "SLA", "U", "V", "WU", "WV"]
FEATURE_CHANNELS = VARS + ["LAT", "LON", "SIN_DOY", "COS_DOY"]

# North Indian Ocean Region of Interest (ROI)
# Bay of Bengal + Arabian Sea (~5-25°N, 45-100°E)
ROI = {
    "lat_min": 5.0,
    "lat_max": 25.0,
    "lon_min": 45.0,
    "lon_max": 100.0,
    "resolution": 0.25,
}

# Sub-regions for regional validation
SUB_REGIONS = {
    "Full North Indian Ocean": {"lat_min": 5.0, "lat_max": 25.0, "lon_min": 45.0, "lon_max": 100.0},
    "Arabian Sea": {"lat_min": 5.0, "lat_max": 25.0, "lon_min": 45.0, "lon_max": 77.0},
    "Bay of Bengal": {"lat_min": 5.0, "lat_max": 25.0, "lon_min": 77.0, "lon_max": 100.0},
}

# Grid dimensions (padded to multiples of 4 for U-Net pooling operations)
RAW_H, RAW_W = 81, 221  # (25-5)/0.25 + 1 = 81, (100-45)/0.25 + 1 = 221
# Padded dimensions divisible by 4
PAD_H, PAD_W = 84, 224

# Default training hyperparameters
DEFAULT_HP = {
    "epochs": 15,
    "n_train": 400,
    "batch_size": 16,
    "learning_rate": 2e-3,
    "weight_decay": 1e-4,
    "embed_dim": 64,
}
