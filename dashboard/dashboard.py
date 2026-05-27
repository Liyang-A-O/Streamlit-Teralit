import json
import os
import random
from math import erf, sqrt
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from PIL import Image, ImageDraw

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Teralit",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global style ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2130;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label { color: #94a3b8 !important; font-size: 0.78rem; letter-spacing: .06em; text-transform: uppercase; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #1a1f2e;
    border: 1px solid #2d3348;
    border-radius: 12px;
    padding: 1rem 1.2rem !important;
}
[data-testid="metric-container"] label { color: #64748b !important; font-size: 0.72rem !important; letter-spacing:.08em; text-transform:uppercase; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #f1f5f9 !important; font-family:'DM Mono',monospace; font-size:1.8rem !important; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

/* Section headers */
.sec-header {
    display: flex; align-items: center; gap: 10px;
    margin: 2rem 0 1rem;
    padding-bottom: .5rem;
    border-bottom: 1px solid #2d3348;
}
.sec-header .icon { font-size: 1.2rem; }
.sec-header h2 { margin:0; font-size:1.1rem; font-weight:600; color:#e2e8f0; letter-spacing:-.01em; }

/* Insight cards */
.insight-card {
    background: linear-gradient(135deg, #1a1f2e 0%, #141824 100%);
    border: 1px solid #2d3348;
    border-left: 3px solid #38bdf8;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: .75rem 0;
    font-size: .875rem;
    color: #cbd5e1;
    line-height: 1.6;
}
.insight-card.warn { border-left-color: #f59e0b; }
.insight-card.ok   { border-left-color: #34d399; }
.insight-card strong { color: #f1f5f9; }

/* Answer box */
.answer-box {
    background: #141824;
    border: 1px solid #2d3348;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.answer-box .q { font-size:.72rem; text-transform:uppercase; letter-spacing:.1em; color:#64748b; margin-bottom:.5rem; }
.answer-box .a { font-size:.95rem; color:#e2e8f0; line-height:1.65; }
.answer-box .rec { margin-top:.75rem; font-size:.82rem; color:#38bdf8; }

/* Tag pills */
.tag {
    display:inline-block; padding:.2rem .65rem; border-radius:99px;
    font-size:.72rem; font-weight:600; margin-right:.3rem;
    font-family:'DM Mono',monospace;
}
.tag-blue  { background:#1e3a5f; color:#7dd3fc; }
.tag-green { background:#14432a; color:#6ee7b7; }
.tag-red   { background:#4c1d1d; color:#fca5a5; }
.tag-amber { background:#422006; color:#fcd34d; }

/* Image gallery card */
.img-caption {
    font-size:.7rem; color:#64748b; font-family:'DM Mono',monospace;
    margin-top:.3rem; text-align:center; word-break:break-all;
}

/* Divider */
.hline { border:none; border-top:1px solid #2d3348; margin:2rem 0; }

/* Scrollable dataframe */
[data-testid="stDataFrame"] { border-radius:8px; overflow:hidden; }

body { background:#0d1117; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark theme ──────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#141824",
    "axes.facecolor":    "#141824",
    "axes.edgecolor":    "#2d3348",
    "axes.labelcolor":   "#94a3b8",
    "axes.titlecolor":   "#e2e8f0",
    "xtick.color":       "#64748b",
    "ytick.color":       "#64748b",
    "grid.color":        "#1e2130",
    "text.color":        "#e2e8f0",
    "figure.dpi":        130,
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})
PALETTE = ["#38bdf8","#34d399","#f59e0b","#f87171","#a78bfa","#fb923c","#e879f9","#4ade80"]

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
OUTPUT_DIR    = BASE_DIR / "output"
COCO_OUTPUT   = OUTPUT_DIR / "_annotations.coco.json"
TRAIN_DIR     = BASE_DIR / "Skin_desease_(Perbaikan)_dataset.coco" / "train"
ORIG_COCO     = TRAIN_DIR / "_annotations.coco.json"
CHART_DIR     = BASE_DIR  # root, where .png charts are saved by notebook

# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_coco(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def parse_coco(coco: dict, train_dir: str):
    images_df      = pd.DataFrame(coco.get("images", []))
    annotations_df = pd.DataFrame(coco.get("annotations", []))
    categories_df  = pd.DataFrame(coco.get("categories", []))

    if not annotations_df.empty and "bbox" in annotations_df.columns:
        annotations_df[["bbox_x","bbox_y","bbox_w","bbox_h"]] = pd.DataFrame(
            annotations_df["bbox"].apply(lambda b: [float(v) for v in b]).tolist(),
            index=annotations_df.index,
        )
        annotations_df["bbox_area"]    = annotations_df["bbox_w"] * annotations_df["bbox_h"]
        annotations_df["aspect_ratio"] = (annotations_df["bbox_w"] / annotations_df["bbox_h"]).round(3)

    cat_map = dict(zip(categories_df["id"], categories_df["name"]))
    if "category_id" in annotations_df.columns:
        annotations_df["class_name"] = annotations_df["category_id"].map(cat_map)
        used_category_ids = set(annotations_df["category_id"].dropna().unique())
        categories_df = categories_df[categories_df["id"].isin(used_category_ids)].reset_index(drop=True)

    images_df["image_area"] = images_df["width"] * images_df["height"]
    images_df["image_aspect_ratio"] = (images_df["width"] / images_df["height"]).round(3)
    images_df["is_small_image"] = (images_df["width"] < 224) | (images_df["height"] < 224)

    image_width_map = dict(zip(images_df["id"], images_df["width"]))
    image_height_map = dict(zip(images_df["id"], images_df["height"]))
    image_area_map = dict(zip(images_df["id"], images_df["image_area"]))

    if not annotations_df.empty:
        annotations_df["image_width"] = annotations_df["image_id"].map(image_width_map)
        annotations_df["image_height"] = annotations_df["image_id"].map(image_height_map)
        annotations_df["image_area"] = annotations_df["image_id"].map(image_area_map)
        annotations_df["bbox_center_x"] = annotations_df["bbox_x"] + (annotations_df["bbox_w"] / 2)
        annotations_df["bbox_center_y"] = annotations_df["bbox_y"] + (annotations_df["bbox_h"] / 2)
        annotations_df["bbox_x_norm"] = (annotations_df["bbox_x"] / annotations_df["image_width"]).round(4)
        annotations_df["bbox_y_norm"] = (annotations_df["bbox_y"] / annotations_df["image_height"]).round(4)
        annotations_df["bbox_w_norm"] = (annotations_df["bbox_w"] / annotations_df["image_width"]).round(4)
        annotations_df["bbox_h_norm"] = (annotations_df["bbox_h"] / annotations_df["image_height"]).round(4)
        annotations_df["bbox_area_ratio"] = (annotations_df["bbox_area"] / annotations_df["image_area"]).round(4)
        annotations_df["bbox_size_label"] = np.select(
            [
                annotations_df["bbox_area"] < 32 * 32,
                annotations_df["bbox_area"] < 96 * 96,
            ],
            ["small", "medium"],
            default="large",
        )

    images_df["file_path"] = images_df["file_name"].apply(
        lambda fn: str(Path(train_dir) / fn)
    )

    # Merge label into images_df
    first_ann = annotations_df.drop_duplicates("image_id")[["image_id","class_name"]]
    images_df = images_df.merge(first_ann, left_on="id", right_on="image_id", how="left")
    images_df["class_name"] = images_df["class_name"].fillna("Unknown")

    return images_df, annotations_df, categories_df

# ── Load data ─────────────────────────────────────────────────────────────────
coco_path = COCO_OUTPUT if COCO_OUTPUT.exists() else ORIG_COCO

with st.sidebar:
    st.markdown("## 🔬 Teralit EDA")

    if coco_path.exists():
        coco      = load_coco(coco_path)
        train_dir = str(TRAIN_DIR)
    else:
        st.error("File COCO JSON tidak ditemukan.")
        st.stop()

    images_df, annotations_df, categories_df = parse_coco(coco, train_dir)
    class_names = sorted(annotations_df["class_name"].dropna().unique().tolist())

    st.markdown("<hr style='border-color:#2d3348;margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown("### 🎛️ Filter")

    selected_classes = st.multiselect(
        "Pilih Kelas Penyakit",
        options=class_names,
        default=class_names,
    )

    MIN_SIZE = 224
    size_range = st.slider(
        "Rentang Width Gambar (px)",
        min_value=int(images_df["width"].min()),
        max_value=int(images_df["width"].max()),
        value=(int(images_df["width"].min()), int(images_df["width"].max())),
    )

    n_gallery = st.slider("Jumlah gambar per galeri", 4, 24, 12, step=4)

    st.markdown("<hr style='border-color:#2d3348;margin:1rem 0'>", unsafe_allow_html=True)
    info = coco.get("info", {})
    st.markdown(f"**Sumber:** {info.get('url', '-')}")
    st.markdown(f"**Dibuat:** {info.get('date_created', '-')}")
    st.markdown(f"**Versi:** {info.get('version', '-')}")

# ── Apply filters ─────────────────────────────────────────────────────────────
ann_filtered = annotations_df[annotations_df["class_name"].isin(selected_classes)]
img_ids_filtered = set(ann_filtered["image_id"])
img_filtered = images_df[
    (images_df["id"].isin(img_ids_filtered)) &
    (images_df["width"] >= size_range[0]) &
    (images_df["width"] <= size_range[1])
]

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:1.5rem 0 .5rem">
  <p style="font-size:.75rem;color:#64748b;letter-spacing:.12em;text-transform:uppercase;margin:0">
    Analisis Data · Penyakit Kulit
  </p>
  <h1 style="margin:.2rem 0 .3rem;font-size:2rem;font-weight:700;color:#f1f5f9;letter-spacing:-.03em">
    Teralit - Sistem Pendeteksi Penyakit Kulit
  </h1>
  <p style="color:#64748b;font-size:.9rem;margin:0">
    Exploratory Data Analysis · Dataset COCO Format
  </p>
</div>
<hr style="border:none;border-top:1px solid #2d3348;margin:.5rem 0 1.5rem">
""", unsafe_allow_html=True)

# ── KPI Metrics ───────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
total_imgs  = len(img_filtered)
total_ann   = len(ann_filtered)
total_cls   = len(selected_classes)
avg_obj     = total_ann / max(total_imgs, 1)
pct_below   = 100 * len(img_filtered[(img_filtered["width"] < MIN_SIZE) | (img_filtered["height"] < MIN_SIZE)]) / max(total_imgs, 1)

c1.metric("Total Gambar",    f"{total_imgs:,}")
c2.metric("Total Anotasi",   f"{total_ann:,}")
c3.metric("Jumlah Kelas",    f"{total_cls}")
c4.metric("Avg Obj/Gambar",  f"{avg_obj:.2f}")
c5.metric(f"Gambar < {MIN_SIZE}px", f"{pct_below:.1f}%", delta="perlu resize" if pct_below > 0 else "aman", delta_color="inverse" if pct_below > 0 else "normal")

st.markdown("<hr style='border:none;border-top:1px solid #2d3348;margin:1.5rem 0'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_eda, tab_qa, tab_gallery, tab_ab, tab_data = st.tabs([
    "📊 EDA & Visualisasi",
    "💡 Jawaban Pertanyaan Bisnis",
    "🖼️ Galeri Dataset",
    "🧪 A/B Testing",
    "📋 Tabel Data",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EDA & Visualisasi
# ══════════════════════════════════════════════════════════════════════════════
with tab_eda:

    # ── EDA #1 : Distribusi Kelas ─────────────────────────────────────────────
    st.markdown("""<div class="sec-header"><span class="icon">📌</span><h2>EDA #1 — Distribusi Kelas Penyakit Kulit</h2></div>""", unsafe_allow_html=True)

    class_stats = (
        ann_filtered.groupby("class_name")
        .agg(jumlah_anotasi=("id","count"))
        .reset_index()
        .sort_values("jumlah_anotasi", ascending=False)
    )
    class_stats["persentase"] = (class_stats["jumlah_anotasi"] / class_stats["jumlah_anotasi"].sum() * 100).round(2)

    highest = class_stats.iloc[0]
    lowest  = class_stats.iloc[-1]
    ratio   = highest["jumlah_anotasi"] / max(lowest["jumlah_anotasi"], 1)
    THRESHOLD = 3.0

    col_chart, col_pie = st.columns([3, 2])

    with col_chart:
        fig, ax = plt.subplots(figsize=(7, max(3, len(class_stats) * 0.7)))
        colors = [
            "#f87171" if n == highest["class_name"] else
            "#38bdf8" if n == lowest["class_name"]  else "#334155"
            for n in class_stats["class_name"]
        ]
        bars = ax.barh(class_stats["class_name"], class_stats["jumlah_anotasi"], color=colors, edgecolor="#0d1117", height=.6)
        for bar, val in zip(bars, class_stats["jumlah_anotasi"]):
            ax.text(val + 2, bar.get_y() + bar.get_height()/2, str(val), va="center", fontsize=10, fontfamily="monospace", color="#e2e8f0")
        ax.set_xlabel("Jumlah Anotasi")
        ax.set_title("Jumlah Anotasi per Kelas", fontweight="bold")
        ax.invert_yaxis()
        legend_patches = [
            mpatches.Patch(color="#f87171", label="Terbanyak"),
            mpatches.Patch(color="#38bdf8", label="Tersedikit"),
            mpatches.Patch(color="#334155", label="Lainnya"),
        ]
        ax.legend(handles=legend_patches, loc="lower right", framealpha=0.2)
        ax.set_xlim(0, class_stats["jumlah_anotasi"].max() * 1.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_pie:
        fig2, ax2 = plt.subplots(figsize=(4.5, 4.5))
        wedge_colors = PALETTE[:len(class_stats)]
        wedges, texts, autotexts = ax2.pie(
            class_stats["jumlah_anotasi"],
            labels=class_stats["class_name"],
            autopct="%1.1f%%",
            startangle=140,
            colors=wedge_colors,
            wedgeprops=dict(width=0.6, edgecolor="#0d1117", linewidth=1.5),
        )
        for at in autotexts:
            at.set_fontsize(8)
            at.set_color("#0d1117")
            at.set_fontweight("bold")
        ax2.set_title("Proporsi Kelas (%)", fontweight="bold")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # Insight card
    imbalance_status = "warn" if ratio > THRESHOLD else "ok"
    imbalance_label  = f"❌ Imbalance terdeteksi (rasio {ratio:.1f}:1 > {THRESHOLD}:1)" if ratio > THRESHOLD else f"✅ Distribusi aman (rasio {ratio:.2f}:1 ≤ {THRESHOLD}:1)"
    st.markdown(f"""
    <div class="insight-card {imbalance_status}">
      <strong>Kelas terbanyak:</strong> {highest['class_name']} ({highest['jumlah_anotasi']} anotasi, {highest['persentase']:.1f}%) &nbsp;|&nbsp;
      <strong>Tersedikit:</strong> {lowest['class_name']} ({lowest['jumlah_anotasi']} anotasi, {lowest['persentase']:.1f}%)<br>
      <strong>Status:</strong> {imbalance_label}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='hline'></div>", unsafe_allow_html=True)

    # ── EDA #2 : Resolusi Gambar ──────────────────────────────────────────────
    st.markdown("""<div class="sec-header"><span class="icon">📐</span><h2>EDA #2 — Distribusi Resolusi Gambar</h2></div>""", unsafe_allow_html=True)

    unique_sizes = set(zip(img_filtered["width"], img_filtered["height"]))
    below_min    = img_filtered[(img_filtered["width"] < MIN_SIZE) | (img_filtered["height"] < MIN_SIZE)]

    col_s1, col_s2, col_s3 = st.columns([2, 2, 1])

    with col_s1:
        fig, ax = plt.subplots(figsize=(5, 4))
        c_sc = ["#f87171" if (w < MIN_SIZE or h < MIN_SIZE) else "#38bdf8"
                for w, h in zip(img_filtered["width"], img_filtered["height"])]
        ax.scatter(img_filtered["width"], img_filtered["height"], c=c_sc, alpha=0.4, s=8)
        ax.axvline(MIN_SIZE, color="#f59e0b", linestyle="--", linewidth=1, label=f"{MIN_SIZE}px")
        ax.axhline(MIN_SIZE, color="#f59e0b", linestyle="--", linewidth=1)
        ax.set_xlabel("Width (px)")
        ax.set_ylabel("Height (px)")
        ax.set_title("Scatter: Width vs Height", fontweight="bold")
        ax.legend(framealpha=0.2, fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_s2:
        fig, axes = plt.subplots(2, 1, figsize=(5, 4))
        axes[0].hist(img_filtered["width"].dropna(), bins=30, color="#38bdf8", edgecolor="#0d1117", alpha=.9)
        axes[0].axvline(MIN_SIZE, color="#f59e0b", linestyle="--", linewidth=1)
        axes[0].set_title("Distribusi Width", fontweight="bold")
        axes[0].set_xlabel("Width (px)")
        axes[1].hist(img_filtered["height"].dropna(), bins=30, color="#34d399", edgecolor="#0d1117", alpha=.9)
        axes[1].axvline(MIN_SIZE, color="#f59e0b", linestyle="--", linewidth=1)
        axes[1].set_title("Distribusi Height", fontweight="bold")
        axes[1].set_xlabel("Height (px)")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_s3:
        st.markdown("**Statistik Dimensi**")
        st.dataframe(
            img_filtered[["width","height"]].describe().round(1).rename(columns={"width":"W","height":"H"}),
            use_container_width=True,
        )
        st.markdown(f"""
        <div style="font-size:.78rem;color:#94a3b8;line-height:1.8;margin-top:.5rem">
          <b>Ukuran unik:</b> {len(unique_sizes)}<br>
          <b>Seragam:</b> {'Ya ✅' if len(unique_sizes)==1 else 'Tidak ❌'}<br>
          <b>&lt;{MIN_SIZE}px:</b> {len(below_min)} ({100*len(below_min)/max(total_imgs,1):.1f}%)
        </div>""", unsafe_allow_html=True)

    resize_rec = "640×640 (YOLOv8) atau 512×512 (EfficientDet)"
    st.markdown(f"""
    <div class="insight-card {'warn' if len(unique_sizes)>1 else 'ok'}">
      Dataset memiliki <strong>{len(unique_sizes)} variasi ukuran</strong> yang berbeda — heterogen dalam resolusi.
      {'⚠️ Wajib resize sebelum training. Rekomendasi: <strong>' + resize_rec + '</strong> dengan letterboxing.' if len(unique_sizes)>1 else '✅ Ukuran seragam, tidak perlu resize.'}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='hline'></div>", unsafe_allow_html=True)

    # ── EDA #3 : Bounding Box ─────────────────────────────────────────────────
    st.markdown("""<div class="sec-header"><span class="icon">📦</span><h2>EDA #3 — Analisis Ukuran Bounding Box</h2></div>""", unsafe_allow_html=True)

    SMALL  = 32 * 32
    MEDIUM = 96 * 96

    small_bb  = ann_filtered[ann_filtered["bbox_area"] < SMALL]
    medium_bb = ann_filtered[(ann_filtered["bbox_area"] >= SMALL) & (ann_filtered["bbox_area"] < MEDIUM)]
    large_bb  = ann_filtered[ann_filtered["bbox_area"] >= MEDIUM]

    col_b1, col_b2 = st.columns(2)

    with col_b1:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.hist(np.log1p(ann_filtered["bbox_area"].dropna()), bins=40, color="#a78bfa", edgecolor="#0d1117", alpha=.9)
        ax.set_xlabel("log(Area + 1)")
        ax.set_title("Distribusi Luas Bbox (Log Scale)", fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b2:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        sizes = [len(small_bb), len(medium_bb), len(large_bb)]
        labels = [f"Kecil\n(<32×32)", f"Sedang\n(32–96px)", f"Besar\n(>96×96)"]
        ax.pie(sizes, labels=labels, autopct="%1.1f%%",
               colors=["#f87171","#f59e0b","#34d399"],
               wedgeprops=dict(width=0.6, edgecolor="#0d1117"),
               startangle=90)
        ax.set_title("Proporsi Ukuran Objek", fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    mean_area = ann_filtered["bbox_area"].mean()
    std_area  = ann_filtered["bbox_area"].std()
    ratio_std = std_area / max(mean_area, 1)

    col_bb1, col_bb2, col_bb3 = st.columns(3)
    col_bb1.metric("Kecil (<32×32 px²)",  f"{len(small_bb):,}",  f"{100*len(small_bb)/max(total_ann,1):.1f}%")
    col_bb2.metric("Sedang (32–96 px)",   f"{len(medium_bb):,}", f"{100*len(medium_bb)/max(total_ann,1):.1f}%")
    col_bb3.metric("Besar (>96×96 px²)",  f"{len(large_bb):,}",  f"{100*len(large_bb)/max(total_ann,1):.1f}%")

    st.markdown(f"""
    <div class="insight-card {'warn' if ratio_std > 1 else 'ok'}">
      <strong>Std/Mean ratio bbox_area: {ratio_std:.2f}×</strong> —
      {'⚠️ Std > Mean → variasi ukuran sangat tinggi. Gunakan arsitektur dengan <strong>Feature Pyramid Network (FPN)</strong>.' if ratio_std > 1 else '✅ Distribusi ukuran bbox relatif normal.'}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='hline'></div>", unsafe_allow_html=True)

    # ── EDA #4 : Validitas Anotasi ────────────────────────────────────────────
    st.markdown("""<div class="sec-header"><span class="icon">✅</span><h2>EDA #4 — Kelengkapan & Validitas Anotasi</h2></div>""", unsafe_allow_html=True)

    all_ids     = set(images_df["id"])
    ann_ids     = set(annotations_df["image_id"])
    no_ann_ids  = all_ids - ann_ids
    invalid_mask = (
        (annotations_df["bbox_x"] < 0) |
        (annotations_df["bbox_y"] < 0) |
        (annotations_df["bbox_w"] <= 0) |
        (annotations_df["bbox_h"] <= 0)
    )
    img_size_map = dict(zip(images_df["id"], zip(images_df["width"], images_df["height"])))

    def is_oob(row):
        iid = row["image_id"]
        if iid not in img_size_map:
            return False
        iw, ih = img_size_map[iid]
        return (row["bbox_x"] + row["bbox_w"] > iw) or (row["bbox_y"] + row["bbox_h"] > ih)

    oob_count     = annotations_df.apply(is_oob, axis=1).sum()
    invalid_count = invalid_mask.sum()
    dup_ann       = annotations_df.duplicated(subset=["id"]).sum()

    val_df = pd.DataFrame({
        "Pemeriksaan": [
            "Gambar tanpa anotasi",
            "Koordinat bbox negatif/nol",
            "Bbox melampaui batas gambar",
            "Duplikasi annotation ID",
        ],
        "Jumlah Masalah": [len(no_ann_ids), invalid_count, oob_count, dup_ann],
    })
    val_df["Status"] = val_df["Jumlah Masalah"].apply(lambda x: "✅ OK" if x == 0 else f"❌ {x} masalah")

    col_v1, col_v2 = st.columns([2, 3])

    with col_v1:
        st.dataframe(val_df, use_container_width=True, hide_index=True)

    with col_v2:
        fig, ax = plt.subplots(figsize=(6, 3))
        colors_v = ["#34d399" if v == 0 else "#f87171" for v in val_df["Jumlah Masalah"]]
        bars = ax.bar(val_df["Pemeriksaan"].str[:20], val_df["Jumlah Masalah"], color=colors_v, edgecolor="#0d1117")
        for bar, val in zip(bars, val_df["Jumlah Masalah"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    str(val), ha="center", fontweight="bold", fontsize=11)
        ax.set_ylabel("Jumlah Masalah")
        ax.set_title("Ringkasan Kualitas Anotasi", fontweight="bold")
        ax.set_ylim(0, max(max(val_df["Jumlah Masalah"]), 1) * 1.5)
        plt.xticks(rotation=20, ha="right", fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    total_issues = val_df["Jumlah Masalah"].sum()
    st.markdown(f"""
    <div class="insight-card {'ok' if total_issues==0 else 'warn'}">
      {'✅ <strong>Dataset valid sepenuhnya.</strong> Tidak ditemukan masalah kualitas pada anotasi.' if total_issues==0
       else f'⚠️ Ditemukan <strong>{total_issues} total masalah</strong> pada anotasi. Lakukan pembersihan sebelum training.'}
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Jawaban Pertanyaan Bisnis
# ══════════════════════════════════════════════════════════════════════════════
with tab_qa:
    st.markdown("""<div class="sec-header"><span class="icon">💡</span><h2>Jawaban Pertanyaan Bisnis</h2></div>""", unsafe_allow_html=True)

    # Q1
    ratio_val   = class_stats.iloc[0]["jumlah_anotasi"] / max(class_stats.iloc[-1]["jumlah_anotasi"], 1) if not class_stats.empty else 0
    imbalance   = ratio_val > 3.0
    tag_q1      = f'<span class="tag tag-{"red" if imbalance else "green"}">{"⚠️ Imbalance" if imbalance else "✅ Seimbang"}</span>'
    rec_q1      = "Gunakan **Data Augmentation** pada kelas minoritas dan/atau `class_weight='balanced'` saat training. Evaluasi per kelas (precision, recall, F1) wajib dilakukan." if imbalance else "Distribusi aman. Tidak perlu teknik balancing khusus, namun tetap pantau metrik per kelas."

    st.markdown(f"""
    <div class="answer-box">
      <div class="q">Pertanyaan 1</div>
      <div class="a">
        Bagaimana distribusi jumlah citra pada setiap kelas penyakit kulit, dan apakah rasio antara kelas terbanyak dan tersedikit menunjukkan adanya <em>class imbalance</em> yang signifikan?
      </div>
      <hr style="border-color:#2d3348;margin:.75rem 0">
      <div style="font-size:.875rem;color:#e2e8f0;line-height:1.7">
        {tag_q1}
        Dari <strong>{len(class_stats)} kelas</strong> yang teranalisis, kelas terbanyak adalah
        <strong>{class_stats.iloc[0]['class_name'] if not class_stats.empty else '-'}</strong>
        ({int(class_stats.iloc[0]['jumlah_anotasi']) if not class_stats.empty else 0} anotasi) dan tersedikit
        <strong>{class_stats.iloc[-1]['class_name'] if not class_stats.empty else '-'}</strong>
        ({int(class_stats.iloc[-1]['jumlah_anotasi']) if not class_stats.empty else 0} anotasi).
        Rasio max/min = <strong>{ratio_val:.2f}:1</strong>
        {'— melebihi ambang batas 3:1 sehingga <strong>class imbalance signifikan terdeteksi</strong>.' if imbalance else '— di bawah ambang batas 3:1, sehingga dataset dikategorikan <strong>seimbang</strong>.'}
      </div>
      <div class="rec">→ {rec_q1}</div>
    </div>
    """, unsafe_allow_html=True)

    # Q2
    n_unique   = len(set(zip(images_df["width"], images_df["height"])))
    pct_b      = 100 * len(images_df[(images_df["width"] < MIN_SIZE) | (images_df["height"] < MIN_SIZE)]) / max(len(images_df), 1)
    tag_q2     = f'<span class="tag tag-{"amber" if n_unique > 1 else "green"}">{"⚠️ Heterogen" if n_unique > 1 else "✅ Seragam"}</span>'

    st.markdown(f"""
    <div class="answer-box">
      <div class="q">Pertanyaan 2</div>
      <div class="a">
        Apakah seluruh gambar dalam dataset memiliki resolusi yang seragam, dan berapa persentase gambar yang resolusinya di bawah 224×224 piksel?
      </div>
      <hr style="border-color:#2d3348;margin:.75rem 0">
      <div style="font-size:.875rem;color:#e2e8f0;line-height:1.7">
        {tag_q2}
        Dataset memiliki <strong>{n_unique} variasi ukuran gambar</strong> yang berbeda — bersifat heterogen.
        Sebanyak <strong>{pct_b:.1f}%</strong> gambar berukuran di bawah {MIN_SIZE}×{MIN_SIZE} px (standar minimum CNN).
        Kondisi ini merupakan konsekuensi pengumpulan gambar dari berbagai sumber klinis yang berbeda.
        Seluruh gambar <strong>wajib di-resize</strong> ke dimensi seragam sebelum dimasukkan ke pipeline training.
      </div>
      <div class="rec">→ Resize ke <strong>640×640</strong> dengan letterboxing (YOLOv8) atau <strong>512×512</strong> (EfficientDet) untuk menjaga aspek rasio asli gambar.</div>
    </div>
    """, unsafe_allow_html=True)

    # Rekomendasi Umum
    st.markdown("""<div class="sec-header" style="margin-top:2rem"><span class="icon">🚀</span><h2>Rekomendasi Action Item untuk AI Engineer</h2></div>""", unsafe_allow_html=True)

    recs = [
        ("🔁", "Data Augmentation", "Terapkan augmentasi (flip, rotate, brightness) pada kelas minoritas untuk menyeimbangkan distribusi training.", "blue"),
        ("📏", "Standardisasi Ukuran", "Resize semua gambar ke 640×640 dengan letterboxing. Gunakan Albumentations atau torchvision transforms.", "green"),
        ("⚖️", "Class Weights", "Gunakan class_weight='balanced' atau focal loss saat training untuk mengatasi imbalance.", "amber"),
        ("🏗️", "Arsitektur FPN", "Pilih arsitektur dengan Feature Pyramid Network (YOLOv8, Faster R-CNN+FPN) untuk menangani variasi ukuran bbox.", "blue"),
        ("📊", "Evaluasi Per Kelas", "Pantau precision, recall, dan F1-score per kelas pada setiap epoch evaluasi — jangan hanya mAP global.", "green"),
    ]

    for icon, title, desc, color in recs:
        st.markdown(f"""
        <div class="insight-card">
          <strong>{icon} {title}</strong><br>
          <span style="color:#94a3b8">{desc}</span>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Galeri Dataset
# ══════════════════════════════════════════════════════════════════════════════
with tab_gallery:
    st.markdown("""<div class="sec-header"><span class="icon">🖼️</span><h2>Galeri Dataset per Kelas</h2></div>""", unsafe_allow_html=True)

    # Filter per kelas
    col_gf1, col_gf2 = st.columns([2, 3])
    with col_gf1:
        gallery_class = st.selectbox("Pilih Kelas", options=["Semua Kelas"] + class_names)
    with col_gf2:
        show_bbox = st.toggle("Tampilkan Bounding Box", value=True)

    # Filter images
    if gallery_class == "Semua Kelas":
        pool = img_filtered.copy()
    else:
        class_img_ids = set(ann_filtered[ann_filtered["class_name"] == gallery_class]["image_id"])
        pool = img_filtered[img_filtered["id"].isin(class_img_ids)]

    available = pool[pool["file_path"].apply(os.path.exists)]

    if available.empty:
        st.warning("Tidak ada gambar yang tersedia di path yang terdeteksi. Pastikan folder `train/` berada di lokasi yang benar.")
    else:
        sample = available.sample(min(n_gallery, len(available)), random_state=42)

        # Build bbox lookup
        ann_by_img = annotations_df.groupby("image_id")

        cols_per_row = 4
        for row_start in range(0, len(sample), cols_per_row):
            row_imgs = sample.iloc[row_start : row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, (_, img_row) in zip(cols, row_imgs.iterrows()):
                with col:
                    try:
                        img = Image.open(img_row["file_path"]).convert("RGB")

                        if show_bbox and img_row["id"] in ann_by_img.groups:
                            draw      = ImageDraw.Draw(img)
                            img_anns  = ann_by_img.get_group(img_row["id"])
                            cls_colors = {
                                name: color
                                for name, color in zip(class_names, ["#f87171","#34d399","#38bdf8","#f59e0b","#a78bfa","#fb923c"])
                            }
                            for _, ann in img_anns.iterrows():
                                x, y, w, h = ann["bbox_x"], ann["bbox_y"], ann["bbox_w"], ann["bbox_h"]
                                c = cls_colors.get(ann.get("class_name",""), "#ffffff")
                                draw.rectangle([x, y, x+w, y+h], outline=c, width=3)
                                draw.rectangle([x, y-16, x+w, y], fill=c)
                                draw.text((x+3, y-14), str(ann.get("class_name","")), fill="#000000")

                        st.image(img, use_column_width=True)

                        class_label = img_row.get("class_name", "-")
                        st.markdown(
                            f'<div class="img-caption">'
                            f'<span class="tag tag-blue">{class_label}</span><br>'
                            f'{img_row["width"]}×{img_row["height"]}px'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    except Exception as e:
                        st.warning(f"Gagal load: {img_row.get('file_name','')}")

        st.markdown(f"<p style='color:#64748b;font-size:.78rem;margin-top:1rem'>Menampilkan {len(sample)} dari {len(available)} gambar tersedia untuk kelas <strong>{gallery_class}</strong></p>", unsafe_allow_html=True)

        # Stats per selected class
        if gallery_class != "Semua Kelas":
            st.markdown("<div class='hline'></div>", unsafe_allow_html=True)
            st.markdown(f"**Statistik Kelas: {gallery_class}**")
            cls_ann = ann_filtered[ann_filtered["class_name"] == gallery_class]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Gambar",   len(pool))
            c2.metric("Total Anotasi",  len(cls_ann))
            c3.metric("Avg Bbox Area",  f"{cls_ann['bbox_area'].mean():.0f} px²" if not cls_ann.empty else "-")
            c4.metric("Avg Aspect Ratio", f"{cls_ann['aspect_ratio'].mean():.2f}" if not cls_ann.empty else "-")

            if not cls_ann.empty:
                fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
                axes[0].hist(np.log1p(cls_ann["bbox_area"].dropna()), bins=25, color="#38bdf8", edgecolor="#0d1117")
                axes[0].set_title(f"Distribusi Bbox Area — {gallery_class}", fontweight="bold")
                axes[0].set_xlabel("log(Area + 1)")
                axes[1].hist(cls_ann["aspect_ratio"].dropna(), bins=25, color="#a78bfa", edgecolor="#0d1117")
                axes[1].set_title(f"Distribusi Aspect Ratio — {gallery_class}", fontweight="bold")
                axes[1].set_xlabel("Aspect Ratio")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — A/B Testing
# ══════════════════════════════════════════════════════════════════════════════
with tab_ab:
    st.markdown("""<div class="sec-header"><span class="icon">🧪</span><h2>A/B Testing Model</h2></div>""", unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-card ok">
      Gunakan panel ini untuk membandingkan dua versi model atau dua pipeline preprocessing.
      Masukkan jumlah sampel evaluasi dan jumlah prediksi benar untuk masing-masing variant.
    </div>
    """, unsafe_allow_html=True)

    def normal_cdf(value):
        return 0.5 * (1 + erf(value / sqrt(2)))

    def proportion_ab_test(success_a, total_a, success_b, total_b):
        rate_a = success_a / total_a if total_a else 0
        rate_b = success_b / total_b if total_b else 0
        pooled = (success_a + success_b) / (total_a + total_b)
        se = sqrt(pooled * (1 - pooled) * ((1 / total_a) + (1 / total_b))) if total_a and total_b else 0
        z_score = (rate_b - rate_a) / se if se else 0
        p_value = 2 * (1 - normal_cdf(abs(z_score))) if se else 1
        diff = rate_b - rate_a
        ci_se = sqrt((rate_a * (1 - rate_a) / total_a) + (rate_b * (1 - rate_b) / total_b)) if total_a and total_b else 0
        ci_low = diff - 1.96 * ci_se
        ci_high = diff + 1.96 * ci_se
        uplift = (diff / rate_a * 100) if rate_a else 0
        return rate_a, rate_b, diff, uplift, z_score, p_value, ci_low, ci_high

    default_total = min(max(total_imgs, 100), 500)
    left_ab, right_ab = st.columns(2)
    with left_ab:
        st.markdown("**Variant A — Baseline**")
        total_a = st.number_input("Jumlah sampel A", min_value=1, value=int(default_total), step=10)
        success_a = st.number_input("Prediksi benar A", min_value=0, max_value=int(total_a), value=int(total_a * 0.72), step=1)
    with right_ab:
        st.markdown("**Variant B — Eksperimen**")
        total_b = st.number_input("Jumlah sampel B", min_value=1, value=int(default_total), step=10)
        success_b = st.number_input("Prediksi benar B", min_value=0, max_value=int(total_b), value=int(total_b * 0.80), step=1)

    alpha = st.slider("Level signifikansi (alpha)", min_value=0.01, max_value=0.10, value=0.05, step=0.01)

    rate_a, rate_b, diff, uplift, z_score, p_value, ci_low, ci_high = proportion_ab_test(
        success_a, total_a, success_b, total_b
    )
    significant = p_value < alpha

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("A Accuracy", f"{rate_a * 100:.2f}%")
    m2.metric("B Accuracy", f"{rate_b * 100:.2f}%", delta=f"{diff * 100:.2f} pp")
    m3.metric("Uplift B vs A", f"{uplift:.2f}%")
    m4.metric("p-value", f"{p_value:.4f}", delta="signifikan" if significant else "belum signifikan")

    ab_summary = pd.DataFrame({
        "Variant": ["A - Baseline", "B - Eksperimen"],
        "Total Sampel": [total_a, total_b],
        "Prediksi Benar": [success_a, success_b],
        "Prediksi Salah": [total_a - success_a, total_b - success_b],
        "Accuracy": [rate_a, rate_b],
    })

    col_ab1, col_ab2 = st.columns([1, 1])
    with col_ab1:
        st.dataframe(
            ab_summary.assign(Accuracy=lambda df: (df["Accuracy"] * 100).round(2).astype(str) + "%"),
            use_container_width=True,
            hide_index=True,
        )
    with col_ab2:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        bars = ax.bar(["A", "B"], [rate_a * 100, rate_b * 100], color=["#64748b", "#38bdf8"], edgecolor="#0d1117")
        for bar, value in zip(bars, [rate_a * 100, rate_b * 100]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.5, f"{value:.1f}%", ha="center", fontweight="bold")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, max(rate_a, rate_b) * 120)
        ax.set_title("Perbandingan Variant A/B", fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown(f"""
    <div class="answer-box">
      <div class="q">Hasil Uji Proporsi Dua Sampel</div>
      <div class="a">
        Selisih accuracy B terhadap A adalah <strong>{diff * 100:.2f} percentage point</strong>
        dengan 95% confidence interval <strong>[{ci_low * 100:.2f}%, {ci_high * 100:.2f}%]</strong>.
        Nilai z-score = <strong>{z_score:.3f}</strong> dan p-value = <strong>{p_value:.4f}</strong>.
      </div>
      <div class="rec">
        {'→ Variant B lebih baik secara statistik pada alpha yang dipilih.' if significant and diff > 0 else '→ Belum ada bukti statistik yang cukup bahwa Variant B lebih baik. Tambah sampel evaluasi atau ulangi eksperimen.'}
      </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Tabel Data
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.markdown("""<div class="sec-header"><span class="icon">📋</span><h2>Tabel Data Lengkap</h2></div>""", unsafe_allow_html=True)

    subtab_imgs, subtab_ann, subtab_cat, subtab_feat, subtab_dict, subtab_valid = st.tabs([
        "🖼️ Images", "📦 Annotations", "🏷️ Categories", "🧠 Feature Engineering", "📘 Data Dictionary", "🔍 Validasi"
    ])

    with subtab_imgs:
        st.markdown(f"**{len(img_filtered)} gambar** (setelah filter)")
        display_cols = [c for c in ["id","file_name","width","height","class_name"] if c in img_filtered.columns]
        st.dataframe(img_filtered[display_cols].reset_index(drop=True), use_container_width=True, height=400)

    with subtab_ann:
        st.markdown(f"**{len(ann_filtered)} anotasi** (setelah filter)")
        display_cols = [c for c in ["id","image_id","class_name","bbox_x","bbox_y","bbox_w","bbox_h","bbox_area","aspect_ratio","bbox_area_ratio","bbox_size_label"] if c in ann_filtered.columns]
        st.dataframe(ann_filtered[display_cols].reset_index(drop=True), use_container_width=True, height=400)

    with subtab_cat:
        display_cols = [c for c in ["id", "name"] if c in categories_df.columns]
        st.dataframe(categories_df[display_cols], use_container_width=True)

    with subtab_feat:
        st.markdown("**Fitur Turunan untuk Kesiapan Modeling**")
        st.markdown("""
        <div class="insight-card ok">
          Fitur turunan dibuat dari metadata gambar dan bounding box agar analisis lebih informatif
          sebelum dataset masuk ke model object detection.
        </div>
        """, unsafe_allow_html=True)

        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Fitur Gambar", "3")
        f2.metric("Fitur Bbox", "8")
        f3.metric("Median Area Ratio", f"{ann_filtered['bbox_area_ratio'].median():.3f}")
        f4.metric("Objek Besar", f"{(ann_filtered['bbox_size_label'] == 'large').mean() * 100:.1f}%")

        feature_cols = [
            "id", "image_id", "class_name",
            "bbox_center_x", "bbox_center_y",
            "bbox_x_norm", "bbox_y_norm", "bbox_w_norm", "bbox_h_norm",
            "bbox_area_ratio", "bbox_size_label",
        ]
        feature_cols = [c for c in feature_cols if c in ann_filtered.columns]
        st.dataframe(ann_filtered[feature_cols].reset_index(drop=True), use_container_width=True, height=360)

        size_dist = (
            ann_filtered["bbox_size_label"]
            .value_counts()
            .rename_axis("bbox_size_label")
            .reset_index(name="jumlah_anotasi")
        )
        st.markdown("**Distribusi Ukuran Objek Hasil Feature Engineering**")
        st.dataframe(size_dist, use_container_width=True, hide_index=True)

    with subtab_dict:
        st.markdown("**Kamus Data Dataset COCO**")
        data_dictionary = pd.DataFrame([
            {"Tabel": "Images", "Kolom": "id", "Tipe": "integer", "Deskripsi": "ID unik untuk setiap gambar."},
            {"Tabel": "Images", "Kolom": "file_name", "Tipe": "text", "Deskripsi": "Nama file gambar di folder train."},
            {"Tabel": "Images", "Kolom": "width", "Tipe": "integer", "Deskripsi": "Lebar gambar dalam piksel."},
            {"Tabel": "Images", "Kolom": "height", "Tipe": "integer", "Deskripsi": "Tinggi gambar dalam piksel."},
            {"Tabel": "Images", "Kolom": "image_area", "Tipe": "integer", "Deskripsi": "Luas gambar, dihitung dari width dikali height."},
            {"Tabel": "Images", "Kolom": "image_aspect_ratio", "Tipe": "float", "Deskripsi": "Rasio lebar terhadap tinggi gambar."},
            {"Tabel": "Images", "Kolom": "is_small_image", "Tipe": "boolean", "Deskripsi": "Penanda gambar dengan width atau height di bawah 224 piksel."},
            {"Tabel": "Images", "Kolom": "class_name", "Tipe": "text", "Deskripsi": "Label kelas utama yang terhubung ke gambar berdasarkan anotasi pertama."},
            {"Tabel": "Annotations", "Kolom": "id", "Tipe": "integer", "Deskripsi": "ID unik untuk setiap anotasi bounding box."},
            {"Tabel": "Annotations", "Kolom": "image_id", "Tipe": "integer", "Deskripsi": "ID gambar yang dirujuk oleh anotasi."},
            {"Tabel": "Annotations", "Kolom": "category_id", "Tipe": "integer", "Deskripsi": "ID kategori penyakit kulit."},
            {"Tabel": "Annotations", "Kolom": "class_name", "Tipe": "text", "Deskripsi": "Nama kelas penyakit kulit, seperti ChickenPox, Eczema, Hives, atau Melanoma."},
            {"Tabel": "Annotations", "Kolom": "bbox_x", "Tipe": "float", "Deskripsi": "Koordinat x kiri atas bounding box."},
            {"Tabel": "Annotations", "Kolom": "bbox_y", "Tipe": "float", "Deskripsi": "Koordinat y kiri atas bounding box."},
            {"Tabel": "Annotations", "Kolom": "bbox_w", "Tipe": "float", "Deskripsi": "Lebar bounding box dalam piksel."},
            {"Tabel": "Annotations", "Kolom": "bbox_h", "Tipe": "float", "Deskripsi": "Tinggi bounding box dalam piksel."},
            {"Tabel": "Annotations", "Kolom": "bbox_area", "Tipe": "float", "Deskripsi": "Luas bounding box, dihitung dari bbox_w dikali bbox_h."},
            {"Tabel": "Annotations", "Kolom": "aspect_ratio", "Tipe": "float", "Deskripsi": "Rasio lebar terhadap tinggi bounding box."},
            {"Tabel": "Annotations", "Kolom": "bbox_center_x", "Tipe": "float", "Deskripsi": "Titik tengah bounding box pada sumbu x."},
            {"Tabel": "Annotations", "Kolom": "bbox_center_y", "Tipe": "float", "Deskripsi": "Titik tengah bounding box pada sumbu y."},
            {"Tabel": "Annotations", "Kolom": "bbox_x_norm", "Tipe": "float", "Deskripsi": "Koordinat x bbox yang dinormalisasi terhadap lebar gambar."},
            {"Tabel": "Annotations", "Kolom": "bbox_y_norm", "Tipe": "float", "Deskripsi": "Koordinat y bbox yang dinormalisasi terhadap tinggi gambar."},
            {"Tabel": "Annotations", "Kolom": "bbox_w_norm", "Tipe": "float", "Deskripsi": "Lebar bbox yang dinormalisasi terhadap lebar gambar."},
            {"Tabel": "Annotations", "Kolom": "bbox_h_norm", "Tipe": "float", "Deskripsi": "Tinggi bbox yang dinormalisasi terhadap tinggi gambar."},
            {"Tabel": "Annotations", "Kolom": "bbox_area_ratio", "Tipe": "float", "Deskripsi": "Proporsi luas bbox terhadap luas gambar."},
            {"Tabel": "Annotations", "Kolom": "bbox_size_label", "Tipe": "text", "Deskripsi": "Kategori ukuran objek: small, medium, atau large."},
            {"Tabel": "Categories", "Kolom": "id", "Tipe": "integer", "Deskripsi": "ID kategori yang dipakai oleh annotation category_id."},
            {"Tabel": "Categories", "Kolom": "name", "Tipe": "text", "Deskripsi": "Nama kategori penyakit kulit yang digunakan sebagai label model."},
        ])
        st.dataframe(data_dictionary, use_container_width=True, hide_index=True, height=520)

    with subtab_valid:
        st.markdown("**Ringkasan Validasi Dataset**")
        val_display = pd.DataFrame({
            "Pemeriksaan": [
                "Total gambar",
                "Total anotasi",
                "Gambar tanpa anotasi",
                "Duplikasi image ID",
                "Duplikasi annotation ID",
                "Duplikasi nama file",
                "Koordinat bbox negatif/nol",
                "Bbox melampaui batas gambar",
                "Missing value (images_df)",
                "Missing value (annotations_df)",
            ],
            "Nilai": [
                len(images_df),
                len(annotations_df),
                len(no_ann_ids),
                images_df.duplicated(subset=["id"]).sum(),
                annotations_df.duplicated(subset=["id"]).sum(),
                images_df.duplicated(subset=["file_name"]).sum(),
                int(invalid_mask.sum()),
                int(oob_count),
                int(images_df[["id","file_name","width","height"]].isnull().sum().sum()),
                int(annotations_df[["id","image_id","category_id","bbox_area"]].isnull().sum().sum()),
            ],
        })
        val_display["Status"] = val_display.apply(
            lambda r: "✅ OK" if (
                (r["Pemeriksaan"].startswith("Total") and r["Nilai"] > 0) or
                (not r["Pemeriksaan"].startswith("Total") and r["Nilai"] == 0)
            ) else f"⚠️ {r['Nilai']}",
            axis=1,
        )
        st.dataframe(val_display, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border:none;border-top:1px solid #2d3348;margin:2.5rem 0 1rem">
<p style="text-align:center;color:#334155;font-size:.75rem;font-family:'DM Mono',monospace">
  Teralit - Sistem Pendeteksi Penyakit Kulit · Dataset COCO Format · Analisis Data Penyakit Kulit
</p>
""", unsafe_allow_html=True)
