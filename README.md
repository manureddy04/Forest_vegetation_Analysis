  # 🌿 Vegetation Loss Analysis

Satellite-based vegetation loss detection using **Google Earth Engine**, **Landsat imagery**, **Random Forest classification**, and **ARIMA forecasting** — visualised through an interactive **Streamlit** dashboard.

---

## Overview

This project compares two Landsat composite periods to detect vegetation loss using:

- **NDVI & EVI** spectral indices from Landsat 5 / 8 (Collection 2, Level 2)
- **Random Forest** classifier with spatial block cross-validation and class balancing
- **ARIMA** time-series model for annual trend forecasting
- **Streamlit** dashboard for interactive exploration of results

---

## Project Structure

```
veg-loss-analysis/
├── pipeline.py        # GEE data fetch + RF training + ARIMA (run in Colab/GEE env)
├── app.py             # Streamlit dashboard (upload outputs to explore)
├── requirements.txt   # Python dependencies
├── outputs/           # Generated .npy arrays + meta.json (git-ignored)
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the pipeline (Google Colab recommended)

> Requires a GEE service account and key file.

1. Upload `pipeline.py` to Google Colab
2. Upload your GEE service account key (`.json`)
3. Update the **USER CONFIGURATION** block at the top of `pipeline.py`:

```python
LATITUDE         = 11.6854       # AOI centre latitude
LONGITUDE        = 76.1320       # AOI centre longitude
BUFFER_KM        = 50            # Radius around the point

DATE_START_BEGIN = "2023-04-01"  # Baseline period start
DATE_START_END   = "2023-05-31"  # Baseline period end
DATE_END_BEGIN   = "2024-07-01"  # Comparison period start
DATE_END_END     = "2024-08-31"  # Comparison period end

SERVICE_ACCOUNT  = "your-service-account@project.iam.gserviceaccount.com"
KEY_FILE         = "/content/your-key.json"
GEE_PROJECT      = "your-gee-project-id"
```

4. Run the notebook — outputs are saved to `/content/veg_outputs/`

### 3. Launch the Streamlit dashboard

```bash
streamlit run app.py
```

Upload files from your `veg_outputs/` folder via the sidebar:

| File | Description |
|------|-------------|
| `ndvi_s.npy` | Baseline NDVI array |
| `ndvi_e.npy` | Comparison NDVI array |
| `ndvi_delta.npy` | NDVI change (Δ) array |
| `evi_s.npy` | Baseline EVI array |
| `evi_e.npy` | Comparison EVI array |
| `evi_delta.npy` | EVI change (Δ) array |
| `pred_binary.npy` | RF binary loss prediction |
| `pred_prob.npy` | RF loss probability map |
| `loss_map.npy` | Ground-truth loss mask |
| `meta.json` | Metadata + model metrics |
| `results.csv` *(optional)* | Summary metrics table |
| `ndvi_plot.png` *(optional)* | Pre-generated NDVI plot |

---

## Dashboard Tabs

| Tab | Contents |
|-----|----------|
| 🗺 NDVI Maps | Baseline, comparison, and Δ NDVI rasters |
| 🌱 EVI Maps | Baseline, comparison, and Δ EVI rasters |
| 🤖 RF Prediction | Binary loss map + probability heatmap |
| 📊 Distributions | NDVI / Δ NDVI histograms with loss threshold |
| 📈 NDVI Plot | Uploaded pre-generated plot image |
| 📋 Results CSV | Tabular model metrics |

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DOWNLOAD_SCALE_M` | 60 | Landsat download resolution (metres) |
| `CLOUD_COVER_MAX` | 80 | Max cloud cover % for scene filtering |
| `LOSS_THRESH_ABS` | -0.10 | Absolute NDVI drop threshold for loss |
| `LOSS_PERCENTILE` | 5 | Percentile threshold (combined with absolute) |
| `PROB_THRESHOLD` | 0.40 | RF probability cut-off for binary loss |
| `RF_N_ESTIMATORS` | 200 | Random Forest tree count |
| `RF_MAX_DEPTH` | 15 | Random Forest max tree depth |
| `ARIMA_YEARS` | 2015–2024 | Years used for ARIMA time-series |

---

## Features

- **Dual-index analysis**: NDVI + EVI for more robust change detection
- **Texture features**: Local std-dev and range for spatial context
- **Spatial block CV**: Avoids spatial autocorrelation leakage in train/test split
- **Balanced training**: Oversampling minority (loss) class for imbalanced data
- **ARIMA forecasting**: 4-year ahead predictions with 95% confidence intervals
- **Retry logic**: Robust GEE download with exponential backoff

---

## Security

> ⚠️ **Never commit your GEE service account key to Git.**

The `.gitignore` excludes all `.json` files (except `sample_meta.json`) and credential files. Store keys securely using environment variables or a secrets manager in production.

---

## License

MIT
