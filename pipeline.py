# ── USER CONFIGURATION
LATITUDE         = 11.6854
LONGITUDE        = 76.1320
BUFFER_KM        = 50

# ── Full date ranges  (YYYY-MM-DD)
# Baseline period  — the "before" composite
DATE_START_BEGIN = "2023-04-01"
DATE_START_END   = "2023-05-31"
# Comparison period — the "after" composite
DATE_END_BEGIN   = "2024-07-01"
DATE_END_END     = "2024-08-31"
#

CLOUD_COVER_MAX  = 80
ARIMA_YEARS      = list(range(2015, 2025))   # years for annual time-series

SERVICE_ACCOUNT  = "statistical@just-genius-423521-t3.iam.gserviceaccount.com"
KEY_FILE         = "/content/just-genius-423521-t3-10797f78983d.json"
GEE_PROJECT      = "just-genius-423521-t3"

DOWNLOAD_SCALE_M    = 60
LOSS_THRESH_ABS     = -0.10
LOSS_PERCENTILE     = 5
RF_N_ESTIMATORS     = 200
RF_MAX_DEPTH        = 15
RF_NO_LOSS_RATIO    = 5
RF_TEST_SIZE        = 0.30
RF_RANDOM_STATE     = 42
BLOCK_SIZE          = 20
PREDICT_CHUNK       = 500_000
PROB_THRESHOLD      = 0.40
N_RETRY             = 3
RETRY_BACKOFF       = 2
OUTPUT_DIR          = "/content/veg_outputs"
# ════════════════════════════════════════════════════════════════════════════

import ee, math, json, os, warnings, requests
from io import BytesIO
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from scipy.ndimage import uniform_filter, maximum_filter, minimum_filter

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score,
                             average_precision_score, precision_recall_curve)
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")
os.makedirs(OUTPUT_DIR, exist_ok=True)

assert -90  < LATITUDE  < 90
assert -180 < LONGITUDE < 180

# ── 1. Authenticate ──────────────────────────────────────────────────────────
credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, KEY_FILE)
ee.Initialize(credentials, project=GEE_PROJECT)
print("✓ GEE initialised")

point = ee.Geometry.Point([LONGITUDE, LATITUDE])
aoi   = point.buffer(BUFFER_KM * 1000).bounds()
print(f"✓ AOI: ({LATITUDE}°, {LONGITUDE}°)  buffer={BUFFER_KM} km")

# ── 2. Named GEE mapper functions ─────────────────────────────────────────────
def add_ndvi_l8(img):
    return img.normalizedDifference(["SR_B5","SR_B4"]).rename("NDVI")

def add_ndvi_l5(img):
    return img.normalizedDifference(["SR_B4","SR_B3"]).rename("NDVI")

def add_evi_l8(img):
    """EVI = 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)"""
    nir  = img.select("SR_B5").multiply(0.0000275).add(-0.2)
    red  = img.select("SR_B4").multiply(0.0000275).add(-0.2)
    blue = img.select("SR_B2").multiply(0.0000275).add(-0.2)
    evi  = nir.subtract(red).multiply(2.5).divide(
           nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)
           ).rename("EVI")
    return evi

def add_evi_l5(img):
    nir  = img.select("SR_B4").multiply(0.0000275).add(-0.2)
    red  = img.select("SR_B3").multiply(0.0000275).add(-0.2)
    blue = img.select("SR_B1").multiply(0.0000275).add(-0.2)
    evi  = nir.subtract(red).multiply(2.5).divide(
           nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)
           ).rename("EVI")
    return evi

def composites(date_begin, date_end):
    """Return (ndvi_img, evi_img) median composites for a given date range."""
    year = int(date_begin[:4])   # used only to pick Landsat sensor
    if year >= 2013:
        col = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
               .filterBounds(aoi).filterDate(date_begin, date_end)
               .filter(ee.Filter.lt("CLOUD_COVER", CLOUD_COVER_MAX)))
        ndvi_col = col.map(add_ndvi_l8)
        evi_col  = col.map(add_evi_l8)
    else:
        col = (ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
               .filterBounds(aoi).filterDate(date_begin, date_end)
               .filter(ee.Filter.lt("CLOUD_COVER", CLOUD_COVER_MAX)))
        ndvi_col = col.map(add_ndvi_l5)
        evi_col  = col.map(add_evi_l5)
    n = col.size().getInfo()
    if n == 0:
        raise RuntimeError(
            f"No scenes for {date_begin}→{date_end}. "
            "Try widening dates or increasing CLOUD_COVER_MAX.")
    return ndvi_col.median().clip(aoi), evi_col.median().clip(aoi), n

def composites_year(year):
    """Convenience wrapper for annual ARIMA time-series."""
    return composites(f"{year}-01-01", f"{year}-12-31")

# ── 3. Download helpers ───────────────────────────────────────────────────────
def _make_session():
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    s = requests.Session()
    r = Retry(total=N_RETRY, backoff_factor=RETRY_BACKOFF,
              status_forcelist=[429,500,502,503,504], allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=r))
    return s

_SESSION = _make_session()

def to_numpy(image, band, label=""):
    url  = image.getDownloadURL({"bands":[band],"region":aoi,
                                  "scale":DOWNLOAD_SCALE_M,"format":"NPY"})
    print(f"  ↓ {label} …", end=" ", flush=True)
    resp = _SESSION.get(url, timeout=300)
    resp.raise_for_status()
    raw  = np.load(BytesIO(resp.content))
    arr  = (raw[band] if raw.dtype.names else raw).astype("float32")
    arr[(arr < -1) | (arr > 1)] = np.nan
    print(f"done {arr.shape}")
    return arr

# ── 4. Fetch main date-range pair ─────────────────────────────────────────────
# Derive short labels for titles / filenames
LABEL_S = f"{DATE_START_BEGIN}→{DATE_START_END}"
LABEL_E = f"{DATE_END_BEGIN}→{DATE_END_END}"
# Extract years for backward-compatible display
YEAR_START = int(DATE_START_BEGIN[:4])
YEAR_END   = int(DATE_END_BEGIN[:4])
YEAR_START = int(DATE_START_BEGIN[:4])
YEAR_END   = int(DATE_END_BEGIN[:4])

assert YEAR_START < YEAR_END

print(f"\nFetching composites …")
print(f"  Baseline  : {LABEL_S}")
print(f"  Comparison: {LABEL_E}")
ndvi_img_s, evi_img_s, n_s = composites(DATE_START_BEGIN, DATE_START_END)
ndvi_img_e, evi_img_e, n_e = composites(DATE_END_BEGIN,   DATE_END_END)
print(f"  Baseline scenes   : {n_s}")
print(f"  Comparison scenes : {n_e}")

print("\nDownloading NDVI & EVI arrays …")
ndvi_s = to_numpy(ndvi_img_s, "NDVI", f"NDVI {YEAR_START}")
ndvi_e = to_numpy(ndvi_img_e, "NDVI", f"NDVI {YEAR_END}")
evi_s  = to_numpy(evi_img_s,  "EVI",  f"EVI  {YEAR_START}")
evi_e  = to_numpy(evi_img_e,  "EVI",  f"EVI  {YEAR_END}")

h = min(ndvi_s.shape[0], ndvi_e.shape[0])
w = min(ndvi_s.shape[1], ndvi_e.shape[1])
ndvi_s, ndvi_e = ndvi_s[:h,:w], ndvi_e[:h,:w]
evi_s,  evi_e  = evi_s[:h,:w],  evi_e[:h,:w]
print(f"✓ Shape: {h}×{w}  ({h*w:,} px)")

# ── 5. Feature engineering ────────────────────────────────────────────────────
def local_std(arr, radius=3):
    filled = np.where(np.isnan(arr), 0.0, arr)
    sz = radius*2+1
    return np.sqrt(np.clip(
        uniform_filter(filled**2,size=sz)-uniform_filter(filled,size=sz)**2,
        0,None)).astype("float32")

def local_range(arr, radius=3):
    filled = np.where(np.isnan(arr), 0.0, arr)
    sz = radius*2+1
    return (maximum_filter(filled,size=sz)-minimum_filter(filled,size=sz)).astype("float32")

ndvi_delta = ndvi_e - ndvi_s      # label only
evi_delta  = evi_e  - evi_s       # supplemental
ndvi_mean  = (ndvi_s + ndvi_e)/2
evi_mean   = (evi_s  + evi_e )/2
texture_s  = local_std(ndvi_s)
texture_e  = local_std(ndvi_e)
ndvi_contrast = local_range(ndvi_s)

feat_names    = ["ndvi_s","ndvi_e","ndvi_mean","evi_s","evi_e","evi_mean",
                 "texture_s","texture_e","ndvi_contrast"]
feature_stack = np.stack(
    [ndvi_s,ndvi_e,ndvi_mean,evi_s,evi_e,evi_mean,
     texture_s,texture_e,ndvi_contrast], axis=-1)
print(f"✓ Features: {feat_names}")

# ── 6. Loss map ───────────────────────────────────────────────────────────────
valid       = ~np.isnan(ndvi_delta)
pct_thresh  = np.nanpercentile(ndvi_delta, LOSS_PERCENTILE)
loss_thresh = min(pct_thresh, LOSS_THRESH_ABS)
loss_map    = np.where(ndvi_delta < loss_thresh, 1, 0).astype("uint8")
loss_px     = int(np.sum(loss_map[valid]==1))
total_px    = int(np.sum(valid))
px_area_km2 = (DOWNLOAD_SCALE_M**2)/1_000_000
loss_area   = loss_px * px_area_km2

print(f"\n{'═'*60}")
print("  NDVI + EVI CHANGE STATISTICS")
print(f"{'═'*60}")
print(f"  Baseline   ({LABEL_S})")
print(f"    NDVI mean : {np.nanmean(ndvi_s):+.4f}  |  EVI mean: {np.nanmean(evi_s):+.4f}")
print(f"  Comparison ({LABEL_E})")
print(f"    NDVI mean : {np.nanmean(ndvi_e):+.4f}  |  EVI mean: {np.nanmean(evi_e):+.4f}")
print(f"  Δ NDVI mean      : {np.nanmean(ndvi_delta):+.4f}  |  Δ EVI mean: {np.nanmean(evi_delta):+.4f}")
print(f"  Loss threshold   : {loss_thresh:+.4f}")
print(f"  Loss pixels      : {loss_px:,} / {total_px:,}  ({loss_px/total_px*100:.2f}%)")
print(f"  Estimated area   : {loss_area:.2f} km²")
print(f"{'═'*60}")

# ── 7. ARIMA time-series on annual NDVI & EVI means ──────────────────────────
print(f"\nBuilding ARIMA time-series ({ARIMA_YEARS[0]}–{ARIMA_YEARS[-1]}) …")
ts_ndvi, ts_evi = [], []

for yr in ARIMA_YEARS:
    try:
        ni, ei, _ = composites_year(yr)
        nm = to_numpy(ni, "NDVI", f"NDVI {yr}")
        em = to_numpy(ei, "EVI",  f"EVI  {yr}")
        ts_ndvi.append(float(np.nanmean(nm)))
        ts_evi.append(float(np.nanmean(em)))
    except Exception as ex:
        print(f"  ⚠ {yr} skipped: {ex}")
        ts_ndvi.append(None)
        ts_evi.append(None)

# Clean None entries
valid_pairs = [(y, n, e) for y,n,e in zip(ARIMA_YEARS,ts_ndvi,ts_evi)
               if n is not None and e is not None]
years_ts   = [v[0] for v in valid_pairs]
ndvi_ts    = np.array([v[1] for v in valid_pairs])
evi_ts     = np.array([v[2] for v in valid_pairs])

FORECAST_STEPS = 4

def fit_arima(series, steps=FORECAST_STEPS):
    model  = ARIMA(series, order=(1,1,1))
    result = model.fit()
    fc     = result.get_forecast(steps=steps)
    return result.fittedvalues, fc.predicted_mean, fc.conf_int()

ndvi_fitted, ndvi_fc, ndvi_ci = fit_arima(ndvi_ts)
evi_fitted,  evi_fc,  evi_ci  = fit_arima(evi_ts)

fc_years = list(range(years_ts[-1]+1, years_ts[-1]+FORECAST_STEPS+1))
print(f"✓ ARIMA fitted on {len(years_ts)} years  |  forecast: {fc_years}")

# ── 8. Spatial block split + balanced RF ─────────────────────────────────────
feat_mask = ~np.isnan(feature_stack).any(axis=-1)
n_bh, n_bw = h//BLOCK_SIZE, w//BLOCK_SIZE
total_blocks = n_bh * n_bw

rng    = np.random.default_rng(RF_RANDOM_STATE)
bids   = np.arange(total_blocks); rng.shuffle(bids)
n_te   = max(1, int(total_blocks * RF_TEST_SIZE))
te_set = set(bids[:n_te]); tr_set = set(bids[n_te:])

train_mask = np.zeros((h,w),dtype=bool)
test_mask  = np.zeros((h,w),dtype=bool)
for bid in range(total_blocks):
    br,bc  = divmod(bid, n_bw)
    r0,r1  = br*BLOCK_SIZE,(br+1)*BLOCK_SIZE
    c0,c1  = bc*BLOCK_SIZE,(bc+1)*BLOCK_SIZE
    if bid in tr_set: train_mask[r0:r1,c0:c1]=True
    else:             test_mask[r0:r1,c0:c1]=True

X_tr_all = feature_stack[train_mask & feat_mask]
y_tr_all = loss_map[train_mask & feat_mask]
X_te     = feature_stack[test_mask  & feat_mask]
y_te     = loss_map[test_mask  & feat_mask]

loss_idx    = np.where(y_tr_all==1)[0]
no_loss_idx = np.where(y_tr_all==0)[0]
n_loss      = len(loss_idx)
n_no_loss   = min(len(no_loss_idx), n_loss * RF_NO_LOSS_RATIO)
sub_no_loss = rng.choice(no_loss_idx, size=n_no_loss, replace=False)
sub_idx     = np.concatenate([loss_idx, sub_no_loss]); rng.shuffle(sub_idx)
X_tr, y_tr  = X_tr_all[sub_idx], y_tr_all[sub_idx]

print(f"\n✓ Balanced: {n_loss:,} loss + {n_no_loss:,} no-loss = {len(X_tr):,} train px")

print("Training RF …", end=" ", flush=True)
rf = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
                             class_weight="balanced", random_state=RF_RANDOM_STATE, n_jobs=-1)
rf.fit(X_tr, y_tr)
print("done")

y_prob = rf.predict_proba(X_te)[:,1]
y_pr   = (y_prob >= PROB_THRESHOLD).astype(int)
acc    = accuracy_score(y_te, y_pr)
cm     = confusion_matrix(y_te, y_pr)
importances = rf.feature_importances_
auc    = roc_auc_score(y_te, y_prob)
ap     = average_precision_score(y_te, y_prob)
precisions, recalls, thresholds_pr = precision_recall_curve(y_te, y_prob)
thresh_idx = np.argmin(np.abs(thresholds_pr - PROB_THRESHOLD))

print(f"\n{'═'*54}")
print("  RANDOM FOREST")
print(f"{'═'*54}")
print(f"  AUC={auc:.4f}  AP={ap:.4f}  acc={acc:.4f}")
print(classification_report(y_te, y_pr, target_names=["No Loss","Loss"], digits=4))

# Full-scene probability + binary prediction
print("Predicting full scene …", end=" ", flush=True)
pred_binary = np.full(loss_map.shape, np.nan, dtype="float32")
pred_prob   = np.full(loss_map.shape, np.nan, dtype="float32")
flat_feat   = feature_stack[feat_mask]
for start in range(0, len(flat_feat), PREDICT_CHUNK):
    chunk  = flat_feat[start:start+PREDICT_CHUNK]
    probs  = rf.predict_proba(chunk)[:,1]
    idx    = np.where(feat_mask.ravel())[0][start:start+PREDICT_CHUNK]
    r,c    = np.unravel_index(idx,(h,w))
    pred_prob[r,c]   = probs
    pred_binary[r,c] = (probs >= PROB_THRESHOLD).astype("float32")
print("done")

# ── 9. Save all arrays for Streamlit ─────────────────────────────────────────
np.save(f"{OUTPUT_DIR}/ndvi_s.npy",      ndvi_s)
np.save(f"{OUTPUT_DIR}/ndvi_e.npy",      ndvi_e)
np.save(f"{OUTPUT_DIR}/evi_s.npy",       evi_s)
np.save(f"{OUTPUT_DIR}/evi_e.npy",       evi_e)
np.save(f"{OUTPUT_DIR}/ndvi_delta.npy",  ndvi_delta)
np.save(f"{OUTPUT_DIR}/evi_delta.npy",   evi_delta)
np.save(f"{OUTPUT_DIR}/loss_map.npy",    loss_map)
np.save(f"{OUTPUT_DIR}/pred_prob.npy",   pred_prob)
np.save(f"{OUTPUT_DIR}/pred_binary.npy", pred_binary)

meta = {
    "latitude": LATITUDE, "longitude": LONGITUDE,
    "buffer_km": BUFFER_KM,
    "year_start": YEAR_START, "year_end": YEAR_END,
    "date_start_begin": DATE_START_BEGIN, "date_start_end": DATE_START_END,
    "date_end_begin":   DATE_END_BEGIN,   "date_end_end":   DATE_END_END,
    "label_s": LABEL_S, "label_e": LABEL_E,
    "loss_thresh": float(loss_thresh), "loss_px": loss_px,
    "total_px": total_px, "loss_area_km2": float(loss_area),
    "px_area_km2": float(px_area_km2), "scale_m": DOWNLOAD_SCALE_M,
    "auc": float(auc), "ap": float(ap), "acc": float(acc),
    "prob_threshold": PROB_THRESHOLD,
    "ndvi_mean_start": float(np.nanmean(ndvi_s)),
    "ndvi_mean_end":   float(np.nanmean(ndvi_e)),
    "evi_mean_start":  float(np.nanmean(evi_s)),
    "evi_mean_end":    float(np.nanmean(evi_e)),
    "feat_names": feat_names,
    "feat_importances": importances.tolist(),
    "years_ts":   years_ts,
    "ndvi_ts":    ndvi_ts.tolist(),
    "evi_ts":     evi_ts.tolist(),
    "ndvi_fitted": ndvi_fitted.tolist(),
    "evi_fitted":  evi_fitted.tolist(),
    "ndvi_fc":    ndvi_fc.tolist(),
    "evi_fc":     evi_fc.tolist(),
    "ndvi_ci":    np.array(ndvi_ci).tolist(),
    "evi_ci":     np.array(evi_ci).tolist(),
    "fc_years":   fc_years,
    "cm": cm.tolist(),
    "precisions": precisions.tolist(),
    "recalls":    recalls.tolist(),
    "thresh_idx": int(thresh_idx),
}
with open(f"{OUTPUT_DIR}/meta.json","w") as f:
    json.dump(meta, f)
print(f"✓ All arrays + metadata saved → {OUTPUT_DIR}/")

# ── 10. Static matplotlib figure (for report / Colab) ────────────────────────
DARK="#0b0f14"; PANEL="#131920"; MINT="#00e5a0"
RED="#ff4d6d";  GOLD="#f0c040";  TEAL="#00b4d8"
WHITE="#e8edf2";GREY="#7a8a9a";  PURPLE="#b07fff"; ORANGE="#ff9f45"

fig = plt.figure(figsize=(30, 18), facecolor=DARK)
gs  = gridspec.GridSpec(3, 5, figure=fig, hspace=0.45, wspace=0.30,
                        left=0.04, right=0.97, top=0.92, bottom=0.05)

lat_dir = "S" if LATITUDE<0 else "N"
lon_dir = "W" if LONGITUDE<0 else "E"
fig.text(0.5,0.955,
    f"VEGETATION LOSS ANALYSIS  ·  {abs(LATITUDE):.2f}°{lat_dir}, "
    f"{abs(LONGITUDE):.2f}°{lon_dir}",
    ha="center",fontsize=15,fontweight="bold",color=WHITE,fontfamily="monospace")

fig.text(0.5,0.938,
    f"Baseline: {LABEL_S}   →   Comparison: {LABEL_E}",
    ha="center",fontsize=10,color=MINT,fontfamily="monospace")
fig.text(0.5,0.922,
    f"Buffer {BUFFER_KM} km  ·  Landsat {DOWNLOAD_SCALE_M} m  ·  "
    f"Loss={loss_area:.1f} km²  ·  AUC={auc:.3f}  AP={ap:.3f}  "
    f"thresh={PROB_THRESHOLD}  ·  EVI + NDVI + ARIMA",
    ha="center",fontsize=9,color=GREY,fontfamily="monospace")

def img_panel(ax, data, title, cmap, vmin, vmax, unit):
    ax.set_facecolor(PANEL)
    cm_obj = plt.get_cmap(cmap).copy() if isinstance(cmap,str) else cmap
    im = ax.imshow(data,cmap=cm_obj,vmin=vmin,vmax=vmax,
                   interpolation="nearest",aspect="auto")
    ax.set_title(title,color=MINT,fontsize=9,pad=5,
                 fontfamily="monospace",fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor("#2a3540"); sp.set_linewidth(1.2)
    cb = fig.colorbar(im,ax=ax,fraction=0.046,pad=0.03,shrink=0.85)
    cb.set_label(unit,color=GREY,fontsize=7,fontfamily="monospace")
    cb.ax.yaxis.set_tick_params(color=GREY,labelsize=6)
    plt.setp(cb.ax.yaxis.get_ticklabels(),color=GREY,fontfamily="monospace")
    cb.outline.set_edgecolor("#2a3540")

# Row 0: NDVI start | NDVI end | EVI start | EVI end | Δ NDVI
r0 = [fig.add_subplot(gs[0,i]) for i in range(5)]
img_panel(r0[0], ndvi_s,    f"NDVI  {DATE_START_BEGIN}",   "YlGn",   0.0,0.9,"NDVI")
img_panel(r0[1], ndvi_e,    f"NDVI  {DATE_END_BEGIN}",     "YlGn",   0.0,0.9,"NDVI")
img_panel(r0[2], evi_s,     f"EVI   {DATE_START_BEGIN}",   "YlGn",  -0.1,0.7,"EVI")
img_panel(r0[3], evi_e,     f"EVI   {DATE_END_BEGIN}",     "YlGn",  -0.1,0.7,"EVI")
img_panel(r0[4], ndvi_delta,f"Δ NDVI  {DATE_START_BEGIN[:7]}→{DATE_END_BEGIN[:7]}","RdYlGn",-0.4,0.4,"Δ NDVI")

# Row 1: Δ EVI | RF binary | RF probability | PR curve | Confusion matrix
r1 = [fig.add_subplot(gs[1,i]) for i in range(5)]
img_panel(r1[0], evi_delta, f"Δ EVI  {DATE_START_BEGIN[:7]}→{DATE_END_BEGIN[:7]}","RdYlGn",-0.3,0.3,"Δ EVI")

# RF binary + GT
pred_disp = np.where(np.isnan(pred_binary),-1,pred_binary)
r1[1].set_facecolor(PANEL)
r1[1].imshow(pred_disp,cmap=ListedColormap(["#1a2e3a",RED]),
             vmin=0,vmax=1,interpolation="nearest",aspect="auto")
gt_c = np.where(loss_map==1,1.0,np.nan)
r1[1].imshow(gt_c,cmap=ListedColormap([GOLD]),vmin=0,vmax=1,
             interpolation="nearest",aspect="auto",alpha=0.5)
r1[1].set_title(f"RF Binary (p≥{PROB_THRESHOLD}) + GT",color=MINT,
                fontsize=9,pad=5,fontfamily="monospace",fontweight="bold")
r1[1].set_xticks([]); r1[1].set_yticks([])
for sp in r1[1].spines.values(): sp.set_edgecolor("#2a3540"); sp.set_linewidth(1.2)
r1[1].legend(handles=[mpatches.Patch(color="#1a2e3a",label="No Loss"),
                       mpatches.Patch(color=RED,label="Loss (RF)"),
                       mpatches.Patch(color=GOLD,label="Loss (GT)")],
             loc="lower right",fontsize=6,facecolor=DARK,
             edgecolor="#2a3540",labelcolor=WHITE,framealpha=0.9)

# RF probability map
prob_disp = np.where(np.isnan(pred_prob), np.nan, pred_prob)
r1[2].set_facecolor(PANEL)
pcmap = plt.get_cmap("plasma").copy(); pcmap.set_bad(color=PANEL)
im_p = r1[2].imshow(prob_disp,cmap=pcmap,vmin=0,vmax=1,
                    interpolation="nearest",aspect="auto")
r1[2].imshow(gt_c,cmap=ListedColormap([GOLD]),vmin=0,vmax=1,
             interpolation="nearest",aspect="auto",alpha=0.3)
r1[2].set_title("RF Loss Probability",color=MINT,fontsize=9,pad=5,
                fontfamily="monospace",fontweight="bold")
r1[2].set_xticks([]); r1[2].set_yticks([])
for sp in r1[2].spines.values(): sp.set_edgecolor("#2a3540"); sp.set_linewidth(1.2)
cb_p = fig.colorbar(im_p,ax=r1[2],fraction=0.046,pad=0.03,shrink=0.85)
cb_p.set_label("P(Loss)",color=GREY,fontsize=7,fontfamily="monospace")
cb_p.ax.axhline(PROB_THRESHOLD,color=RED,lw=1.5,ls="--")
cb_p.ax.yaxis.set_tick_params(color=GREY,labelsize=6)
plt.setp(cb_p.ax.yaxis.get_ticklabels(),color=GREY,fontfamily="monospace")
cb_p.outline.set_edgecolor("#2a3540")

# PR curve
r1[3].set_facecolor(PANEL)
r1[3].plot(recalls,precisions,color=TEAL,lw=2,label=f"AP={ap:.3f}")
r1[3].axvline(recalls[thresh_idx],color=RED,lw=1.5,ls="--",
              label=f"t={PROB_THRESHOLD} P={precisions[thresh_idx]:.2f} R={recalls[thresh_idx]:.2f}")
r1[3].scatter([recalls[thresh_idx]],[precisions[thresh_idx]],color=ORANGE,s=60,zorder=5)
r1[3].fill_between(recalls,precisions,alpha=0.12,color=TEAL)
r1[3].set_xlim(0,1); r1[3].set_ylim(0,1.05)
r1[3].set_title("Precision-Recall Curve",color=MINT,fontsize=9,pad=5,
                fontfamily="monospace",fontweight="bold")
r1[3].set_xlabel("Recall",color=GREY,fontsize=8,fontfamily="monospace")
r1[3].set_ylabel("Precision",color=GREY,fontsize=8,fontfamily="monospace")
r1[3].tick_params(colors=GREY,labelsize=7)
r1[3].legend(facecolor=DARK,edgecolor="#2a3540",labelcolor=WHITE,fontsize=7,framealpha=0.9)
for sp in r1[3].spines.values(): sp.set_edgecolor("#2a3540")

# Confusion matrix
r1[4].set_facecolor(PANEL)
cm_norm = cm.astype(float)/cm.sum(axis=1,keepdims=True)
r1[4].imshow(cm_norm,cmap="YlOrRd",vmin=0,vmax=1,aspect="auto")
r1[4].set_xticks([0,1]); r1[4].set_yticks([0,1])
r1[4].set_xticklabels(["No Loss","Loss"],color=GREY,fontsize=7,fontfamily="monospace")
r1[4].set_yticklabels(["No Loss","Loss"],color=GREY,fontsize=7,fontfamily="monospace")
r1[4].set_xlabel("Predicted",color=GREY,fontsize=8,fontfamily="monospace")
r1[4].set_ylabel("Actual",color=GREY,fontsize=8,fontfamily="monospace")
r1[4].set_title(f"Confusion Matrix (AUC={auc:.3f})",color=MINT,fontsize=9,pad=5,
                fontfamily="monospace",fontweight="bold")
for i in range(2):
    for j in range(2):
        lbl=["TN","FP","FN","TP"][(i*2)+j]
        r1[4].text(j,i,f"{lbl}\n{cm_norm[i,j]:.2f}\n({cm[i,j]:,})",
                   ha="center",va="center",fontsize=8,
                   color="white" if cm_norm[i,j]>0.6 else DARK,
                   fontfamily="monospace",fontweight="bold")
for sp in r1[4].spines.values(): sp.set_edgecolor("#2a3540")

# Row 2: ARIMA NDVI | ARIMA EVI | Feature importances | NDVI dist | EVI dist
r2 = [fig.add_subplot(gs[2,i]) for i in range(5)]

# ARIMA NDVI
r2[0].set_facecolor(PANEL)
r2[0].plot(years_ts, ndvi_ts, color=MINT, lw=2, marker="o", ms=5, label="Observed NDVI")
r2[0].plot(years_ts, ndvi_fitted, color=TEAL, lw=1.5, ls="--", label="ARIMA fitted")
r2[0].plot(fc_years, ndvi_fc, color=GOLD, lw=2, marker="s", ms=5, label="Forecast")
r2[0].fill_between(fc_years, ndvi_ci[:,0], ndvi_ci[:,1], color=GOLD, alpha=0.2, label="95% CI")
r2[0].axvline(years_ts[-1]+0.5, color=GREY, lw=1, ls=":")
r2[0].set_title("ARIMA Forecast — NDVI", color=MINT, fontsize=9, pad=5,
                fontfamily="monospace", fontweight="bold")
r2[0].set_xlabel("Year", color=GREY, fontsize=8, fontfamily="monospace")
r2[0].set_ylabel("Mean NDVI", color=GREY, fontsize=8, fontfamily="monospace")
r2[0].tick_params(colors=GREY, labelsize=7)
r2[0].legend(facecolor=DARK, edgecolor="#2a3540", labelcolor=WHITE, fontsize=7, framealpha=0.9)
for sp in r2[0].spines.values(): sp.set_edgecolor("#2a3540")

# ARIMA EVI
r2[1].set_facecolor(PANEL)
r2[1].plot(years_ts, evi_ts, color=PURPLE, lw=2, marker="o", ms=5, label="Observed EVI")
r2[1].plot(years_ts, evi_fitted, color=TEAL, lw=1.5, ls="--", label="ARIMA fitted")
r2[1].plot(fc_years, evi_fc, color=ORANGE, lw=2, marker="s", ms=5, label="Forecast")
r2[1].fill_between(fc_years, evi_ci[:,0], evi_ci[:,1], color=ORANGE, alpha=0.2, label="95% CI")
r2[1].axvline(years_ts[-1]+0.5, color=GREY, lw=1, ls=":")
r2[1].set_title("ARIMA Forecast — EVI", color=MINT, fontsize=9, pad=5,
                fontfamily="monospace", fontweight="bold")
r2[1].set_xlabel("Year", color=GREY, fontsize=8, fontfamily="monospace")
r2[1].set_ylabel("Mean EVI", color=GREY, fontsize=8, fontfamily="monospace")
r2[1].tick_params(colors=GREY, labelsize=7)
r2[1].legend(facecolor=DARK, edgecolor="#2a3540", labelcolor=WHITE, fontsize=7, framealpha=0.9)
for sp in r2[1].spines.values(): sp.set_edgecolor("#2a3540")

# Feature importances
r2[2].set_facecolor(PANEL)
sidx = np.argsort(importances)
fc_map = {"ndvi_s":MINT,"ndvi_e":GOLD,"ndvi_mean":TEAL,
          "evi_s":"#88ff88","evi_e":"#ffcc44","evi_mean":"#44ddcc",
          "texture_s":PURPLE,"texture_e":PURPLE,"ndvi_contrast":RED}
bcolors = [fc_map.get(feat_names[i],TEAL) for i in sidx]
r2[2].barh([feat_names[i] for i in sidx], importances[sidx],
           color=bcolors, edgecolor="#2a3540")
for i,(idx,imp) in enumerate(zip(sidx,importances[sidx])):
    r2[2].text(imp+0.002,i,f"{imp:.3f}",va="center",
               color=GREY,fontsize=6,fontfamily="monospace")
r2[2].set_title("RF Feature Importances\n(NDVI + EVI + texture)",
                color=MINT,fontsize=9,pad=5,fontfamily="monospace",fontweight="bold")
r2[2].set_xlabel("Mean Decrease Impurity",color=GREY,fontsize=8,fontfamily="monospace")
r2[2].tick_params(colors=GREY,labelsize=7)
for sp in r2[2].spines.values(): sp.set_edgecolor("#2a3540")

# NDVI distribution
r2[3].set_facecolor(PANEL)
bins = np.linspace(-0.1,1.0,80)
r2[3].hist(ndvi_s[~np.isnan(ndvi_s)].ravel(),bins=bins,alpha=0.6,
           color=MINT,label=f"NDVI {DATE_START_BEGIN[:7]}",density=True)
r2[3].hist(ndvi_e[~np.isnan(ndvi_e)].ravel(),bins=bins,alpha=0.6,
           color=GOLD,label=f"NDVI {DATE_END_BEGIN[:7]}",density=True)
r2[3].axvline(loss_thresh,color=RED,lw=1.5,ls="--",label=f"thresh={loss_thresh:.2f}")
r2[3].set_title("NDVI Distribution",color=MINT,fontsize=9,pad=5,
                fontfamily="monospace",fontweight="bold")
r2[3].set_xlabel("NDVI",color=GREY,fontsize=8,fontfamily="monospace")
r2[3].set_ylabel("Density",color=GREY,fontsize=8,fontfamily="monospace")
r2[3].tick_params(colors=GREY,labelsize=7)
r2[3].legend(facecolor=DARK,edgecolor="#2a3540",labelcolor=WHITE,fontsize=7,framealpha=0.9)
for sp in r2[3].spines.values(): sp.set_edgecolor("#2a3540")

# EVI distribution
r2[4].set_facecolor(PANEL)
bins_e = np.linspace(-0.2,0.8,80)
r2[4].hist(evi_s[~np.isnan(evi_s)].ravel(),bins=bins_e,alpha=0.6,
           color=PURPLE,label=f"EVI {DATE_START_BEGIN[:7]}",density=True)
r2[4].hist(evi_e[~np.isnan(evi_e)].ravel(),bins=bins_e,alpha=0.6,
           color=ORANGE,label=f"EVI {DATE_END_BEGIN[:7]}",density=True)
r2[4].set_title("EVI Distribution",color=MINT,fontsize=9,pad=5,
                fontfamily="monospace",fontweight="bold")
r2[4].set_xlabel("EVI",color=GREY,fontsize=8,fontfamily="monospace")
r2[4].set_ylabel("Density",color=GREY,fontsize=8,fontfamily="monospace")
r2[4].tick_params(colors=GREY,labelsize=7)
r2[4].legend(facecolor=DARK,edgecolor="#2a3540",labelcolor=WHITE,fontsize=7,framealpha=0.9)
for sp in r2[4].spines.values(): sp.set_edgecolor("#2a3540")

out_png = "/content/vegetation_loss_v7.png"
plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.show()
print(f"\n✓ Figure saved → {out_png}")
