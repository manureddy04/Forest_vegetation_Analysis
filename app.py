"""
Vegetation Loss Visualiser — File Upload Version
Upload your .npy, meta.json, results.csv and ndvi_plot.png files.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import json
import io
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Vegetation Loss Visualiser", page_icon="🌿", layout="wide")

DARK  = "#0b0f14"
PANEL = "#131920"
MINT  = "#00e5a0"
RED   = "#ff4d6d"
GOLD  = "#f0c040"
TEAL  = "#00b4d8"
WHITE = "#e8edf2"
GREY  = "#7a8a9a"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌿 Veg Loss Visualiser")
    st.markdown("---")
    st.markdown("Upload all files from your `veg_outputs/` folder below.")
    st.markdown("---")

    uploaded_files = st.file_uploader(
        "Upload files (.npy, .json, .csv, .png)",
        type=["npy", "json", "csv", "png"],
        accept_multiple_files=True,
    )

st.title("🌿 Vegetation Loss Visualiser")
st.markdown("---")

if not uploaded_files:
    st.info("👈 Upload your files from the sidebar to get started.")
    st.markdown("""
**Expected files from `veg_outputs/` folder:**
- `ndvi_s.npy`, `ndvi_e.npy`, `ndvi_delta.npy`
- `evi_s.npy`, `evi_e.npy`, `evi_delta.npy`
- `pred_binary.npy`, `pred_prob.npy`, `loss_map.npy`
- `meta.json`
- `results.csv`
- `ndvi_plot.png`
    """)
    st.stop()

# ── Load uploaded files into a dict by name ───────────────────────────────────
file_dict = {f.name: f for f in uploaded_files}

def load_npy(name):
    if name in file_dict:
        file_dict[name].seek(0)
        return np.load(io.BytesIO(file_dict[name].read())).astype("float32")
    return None

ndvi_s     = load_npy("ndvi_s.npy")
ndvi_e     = load_npy("ndvi_e.npy")
ndvi_delta = load_npy("ndvi_delta.npy")
evi_s      = load_npy("evi_s.npy")
evi_e      = load_npy("evi_e.npy")
evi_delta  = load_npy("evi_delta.npy")
pred_prob  = load_npy("pred_prob.npy")
pred_bin   = load_npy("pred_binary.npy")
loss_map   = load_npy("loss_map.npy")

# Load meta.json
meta = {}
if "meta.json" in file_dict:
    file_dict["meta.json"].seek(0)
    meta = json.load(file_dict["meta.json"])

# Load results.csv
results_df = None
if "results.csv" in file_dict:
    file_dict["results.csv"].seek(0)
    results_df = pd.read_csv(file_dict["results.csv"])

# Load ndvi_plot.png
ndvi_plot_img = None
if "ndvi_plot.png" in file_dict:
    file_dict["ndvi_plot.png"].seek(0)
    ndvi_plot_img = Image.open(io.BytesIO(file_dict["ndvi_plot.png"].read()))

if ndvi_s is None or ndvi_e is None or ndvi_delta is None:
    st.error("Could not find `ndvi_s.npy`, `ndvi_e.npy`, `ndvi_delta.npy`. Please upload all required files.")
    st.stop()

st.success(f"✅ {len(uploaded_files)} file(s) loaded successfully.")

label_s = meta.get("label_s", "Baseline")
label_e = meta.get("label_e", "Comparison")

# ── Metrics ───────────────────────────────────────────────────────────────────
loss_area = meta.get("loss_area_km2", 0)
auc_val   = meta.get("auc", None)
ap_val    = meta.get("ap", None)
acc_val   = meta.get("acc", meta.get("accuracy", None))

# Fall back to results.csv if meta missing values
if results_df is not None:
    rv = dict(zip(results_df["Metric"], results_df["Value"]))
    if auc_val  is None: auc_val  = rv.get("AUC Score")
    if ap_val   is None: ap_val   = rv.get("AP Score")
    if acc_val  is None: acc_val  = rv.get("Model Accuracy")
    if loss_area == 0:   loss_area = rv.get("Vegetation Loss Area", 0)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Loss Area",     f"{float(loss_area):.1f} km²")
c2.metric("Δ NDVI mean",   f"{np.nanmean(ndvi_delta):+.4f}")
c3.metric("ROC-AUC",       f"{float(auc_val):.4f}"  if auc_val  is not None else "—")
c4.metric("Avg Precision", f"{float(ap_val):.4f}"   if ap_val   is not None else "—")
c5.metric("RF Accuracy",   f"{float(acc_val):.4f}"  if acc_val  is not None else "—")
st.markdown("---")

# ── Helper ────────────────────────────────────────────────────────────────────
def img_panel(fig, ax, data, title, cmap, vmin, vmax, unit):
    ax.set_facecolor(PANEL)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest", aspect="auto")
    ax.set_title(title, color=MINT, fontsize=11,
                 fontfamily="monospace", fontweight="bold", pad=6)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#2a3540"); sp.set_linewidth(1.2)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, shrink=0.85)
    cb.set_label(unit, color=GREY, fontsize=8, fontfamily="monospace")
    cb.ax.yaxis.set_tick_params(color=GREY, labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=GREY, fontfamily="monospace")
    cb.outline.set_edgecolor("#2a3540")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs(["🗺 NDVI Maps", "🌱 EVI Maps", "🤖 RF Prediction", "📊 Distributions", "📈 NDVI Plot", "📋 Results CSV"])

with tabs[0]:
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), facecolor=DARK)
    plt.subplots_adjust(wspace=0.25)
    img_panel(fig, axes[0], ndvi_s,     f"NDVI  {label_s}", "YlGn",    0.0,  0.9, "NDVI")
    img_panel(fig, axes[1], ndvi_e,     f"NDVI  {label_e}", "YlGn",    0.0,  0.9, "NDVI")
    img_panel(fig, axes[2], ndvi_delta, "Δ NDVI",           "RdYlGn", -0.4,  0.4, "Δ NDVI")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with tabs[1]:
    if evi_s is not None and evi_e is not None and evi_delta is not None:
        fig, axes = plt.subplots(1, 3, figsize=(21, 6), facecolor=DARK)
        plt.subplots_adjust(wspace=0.25)
        img_panel(fig, axes[0], evi_s,     f"EVI  {label_s}", "YlGn",   -0.1, 0.7, "EVI")
        img_panel(fig, axes[1], evi_e,     f"EVI  {label_e}", "YlGn",   -0.1, 0.7, "EVI")
        img_panel(fig, axes[2], evi_delta, "Δ EVI",           "RdYlGn", -0.3, 0.3, "Δ EVI")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("EVI files not found. Please upload `evi_s.npy`, `evi_e.npy`, `evi_delta.npy`.")

with tabs[2]:
    if pred_bin is not None and pred_prob is not None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=DARK)
        plt.subplots_adjust(wspace=0.2)

        ax = axes[0]; ax.set_facecolor(PANEL)
        ax.imshow(np.where(np.isnan(pred_bin), -1, pred_bin),
                  cmap=ListedColormap(["#1a2e3a", RED]),
                  vmin=0, vmax=1, interpolation="nearest", aspect="auto")
        ax.set_title("RF Predicted Loss", color=MINT, fontsize=11,
                     fontfamily="monospace", fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(handles=[mpatches.Patch(color="#1a2e3a", label="No Loss"),
                            mpatches.Patch(color=RED, label="Loss")],
                  loc="lower right", fontsize=9, facecolor=DARK,
                  edgecolor="#2a3540", labelcolor=WHITE, framealpha=0.9)
        for sp in ax.spines.values(): sp.set_edgecolor("#2a3540")

        ax = axes[1]; ax.set_facecolor(PANEL)
        pcmap = plt.get_cmap("plasma").copy(); pcmap.set_bad(color=PANEL)
        im = ax.imshow(pred_prob, cmap=pcmap, vmin=0, vmax=1,
                       interpolation="nearest", aspect="auto")
        ax.set_title("RF Loss Probability", color=MINT, fontsize=11,
                     fontfamily="monospace", fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, shrink=0.85)
        cb.set_label("P(Loss)", color=GREY, fontsize=8, fontfamily="monospace")
        cb.outline.set_edgecolor("#2a3540")
        for sp in ax.spines.values(): sp.set_edgecolor("#2a3540")

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("Please upload `pred_binary.npy` and `pred_prob.npy` to see RF predictions.")

with tabs[3]:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor=DARK)
    plt.subplots_adjust(wspace=0.3)

    ax = axes[0]; ax.set_facecolor(PANEL)
    bins = np.linspace(-0.1, 1.0, 80)
    ax.hist(ndvi_s[~np.isnan(ndvi_s)].ravel(), bins=bins, alpha=0.65,
            color=MINT, label=f"NDVI {label_s}", density=True)
    ax.hist(ndvi_e[~np.isnan(ndvi_e)].ravel(), bins=bins, alpha=0.65,
            color=GOLD, label=f"NDVI {label_e}", density=True)
    if meta.get("loss_thresh"):
        ax.axvline(meta["loss_thresh"], color=RED, lw=2, ls="--",
                   label=f"threshold = {meta['loss_thresh']:.2f}")
    ax.set_title("NDVI Distribution", color=MINT, fontsize=11,
                 fontfamily="monospace", fontweight="bold")
    ax.set_xlabel("NDVI", color=GREY, fontsize=9, fontfamily="monospace")
    ax.set_ylabel("Density", color=GREY, fontsize=9, fontfamily="monospace")
    ax.tick_params(colors=GREY)
    ax.legend(facecolor=DARK, edgecolor="#2a3540", labelcolor=WHITE, fontsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor("#2a3540")

    ax = axes[1]; ax.set_facecolor(PANEL)
    valid = ~np.isnan(ndvi_delta)
    ax.hist(ndvi_delta[valid].ravel(), bins=70, color=TEAL, alpha=0.8, density=True)
    if meta.get("loss_thresh"):
        ax.axvline(meta["loss_thresh"], color=RED, lw=2, ls="--",
                   label=f"threshold = {meta['loss_thresh']:.2f}")
    ax.set_title("Δ NDVI Distribution", color=MINT, fontsize=11,
                 fontfamily="monospace", fontweight="bold")
    ax.set_xlabel("Δ NDVI", color=GREY, fontsize=9, fontfamily="monospace")
    ax.set_ylabel("Density", color=GREY, fontsize=9, fontfamily="monospace")
    ax.tick_params(colors=GREY)
    ax.legend(facecolor=DARK, edgecolor="#2a3540", labelcolor=WHITE, fontsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor("#2a3540")

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with tabs[4]:
    if ndvi_plot_img is not None:
        st.image(ndvi_plot_img, caption="NDVI Comparison Plot", use_container_width=True)
    else:
        st.info("Upload `ndvi_plot.png` to see the NDVI comparison plot.")

with tabs[5]:
    if results_df is not None:
        st.markdown("### 📋 Results Summary")
        st.dataframe(
            results_df.style.format({"Value": "{:.6f}"}),
            use_container_width=True,
        )
    else:
        st.info("Upload `results.csv` to see the results table.")
