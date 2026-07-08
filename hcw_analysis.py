"""
Hybrid Constructed Wetland (HCW) - Comprehensive Data Analysis
Dataset: Pune Hybrid Constructed Wetland ML Dataset
Generates:
  1. Dataset overview / descriptive statistics
  2. Correlation matrix (numeric features)
  3. Important parameter distributions
  4. Removal efficiency plots
  5. Seasonal & wetland-type analysis
  6. Influent vs Effluent comparisons
  7. Feature importance via RandomForest
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# ──────────────────────────────────────────
# Config
# ──────────────────────────────────────────
DATA_PATH  = r"C:\Users\SUPRIYA\OneDrive\Desktop\UBTECH\hcw_ml_project\hcw_ml_project\pune_hybrid_constructed_wetland_ml_dataset_28451_v2.xlsx"
OUT_DIR = Path(r"C:\Users\SUPRIYA\OneDrive\Desktop\UBTECH\hcw_ml_project\hcw_ml_project\results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE    = "viridis"
STYLE      = "whitegrid"
FIG_DPI    = 150
sns.set_style(STYLE)
sns.set_palette("tab10")

TARGET     = "overall_treatment_efficiency_pct"
REMOVAL_COLS = [
    "BOD_removal_pct", "COD_removal_pct", "TSS_removal_pct",
    "TN_removal_pct",  "TP_removal_pct",  "fecal_coliform_log_removal",
]
INFLUENT_PARAMS = [
    "influent_BOD_mg_L", "influent_COD_mg_L", "influent_TSS_mg_L",
    "influent_TN_mg_L",  "influent_TP_mg_L",  "influent_pH",
]
EFFLUENT_PARAMS = [
    "effluent_BOD_mg_L", "effluent_COD_mg_L", "effluent_TSS_mg_L",
    "effluent_TN_mg_L",  "effluent_TP_mg_L",  "effluent_pH",
]
DESIGN_PARAMS = [
    "hydraulic_loading_rate_m_day", "hydraulic_retention_time_day",
    "bed_area_m2", "bed_depth_m", "plant_density_plants_m2",
    "flow_m3_day",
]

# ──────────────────────────────────────────
# Load data
# ──────────────────────────────────────────
print("Loading dataset …")
df = pd.read_excel(DATA_PATH)
print(f"  Shape: {df.shape}")

numeric_df = df.select_dtypes(include=[np.number])

# ═══════════════════════════════════════════════════════════════════════════════
# 1 ▸ DATASET OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1/7] Dataset Overview …")
desc = numeric_df.describe().T[["count","mean","std","min","50%","max"]]
desc.columns = ["Count","Mean","Std","Min","Median","Max"]

fig, ax = plt.subplots(figsize=(16, max(6, len(desc)*0.28)))
ax.axis("off")
tbl = ax.table(
    cellText=desc.round(2).values,
    rowLabels=desc.index,
    colLabels=desc.columns,
    cellLoc="center", loc="center",
    bbox=[0, 0, 1, 1],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(7)
for (r, c), cell in tbl.get_celld().items():
    if r == 0 or c == -1:
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor("#ECF0F1" if r % 2 == 0 else "white")
plt.title("Dataset Descriptive Statistics", fontsize=14, fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig(OUT_DIR / "01_dataset_overview.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  ✓ Saved 01_dataset_overview.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 2 ▸ CORRELATION MATRIX
# ═══════════════════════════════════════════════════════════════════════════════
print("[2/7] Correlation Matrix …")

# Focus on the most informative numeric columns
corr_cols = (
    DESIGN_PARAMS +
    ["water_temperature_C", "rainfall_mm_day"] +
    INFLUENT_PARAMS +
    REMOVAL_COLS +
    [TARGET]
)
corr_cols = [c for c in corr_cols if c in numeric_df.columns]
corr = numeric_df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(18, 15))
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(
    corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
    annot=True, fmt=".2f", annot_kws={"size": 7},
    square=True, linewidths=0.4, ax=ax,
    cbar_kws={"shrink": 0.7, "label": "Pearson r"},
)
ax.set_title("Correlation Matrix – Key Parameters", fontsize=15, fontweight="bold", pad=15)
ax.tick_params(axis="x", rotation=45, labelsize=8)
ax.tick_params(axis="y", rotation=0,  labelsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR / "02_correlation_matrix.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  ✓ Saved 02_correlation_matrix.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 3 ▸ REMOVAL EFFICIENCY DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════════════════
print("[3/7] Removal Efficiency Distributions …")
removal_labels = ["BOD", "COD", "TSS", "TN", "TP", "Fecal\nColiform (log)"]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
colors = sns.color_palette("tab10", 6)
for i, (col, lbl, clr) in enumerate(zip(REMOVAL_COLS, removal_labels, colors)):
    if col not in df.columns:
        continue
    data = df[col].dropna()
    axes[i].hist(data, bins=50, color=clr, edgecolor="white", alpha=0.85)
    axes[i].axvline(data.mean(),  color="black",  lw=1.8, ls="--", label=f"Mean: {data.mean():.1f}")
    axes[i].axvline(data.median(), color="crimson", lw=1.5, ls=":",  label=f"Median: {data.median():.1f}")
    axes[i].set_title(f"{lbl} Removal", fontsize=12, fontweight="bold")
    axes[i].set_xlabel("Removal (%)" if "log" not in col else "Log Removal", fontsize=9)
    axes[i].set_ylabel("Frequency", fontsize=9)
    axes[i].legend(fontsize=8)
    axes[i].grid(True, alpha=0.3)

fig.suptitle("Pollutant Removal Efficiency Distributions", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / "03_removal_efficiency_distributions.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  ✓ Saved 03_removal_efficiency_distributions.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 4 ▸ SEASONAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("[4/7] Seasonal Analysis …")
season_order = ["Winter", "Summer", "Monsoon", "Post-monsoon"]
seasons_present = [s for s in season_order if s in df["season"].unique()]

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()
palette_s = sns.color_palette("Set2", len(seasons_present))

for i, col in enumerate(REMOVAL_COLS):
    if col not in df.columns:
        continue
    season_data = [df.loc[df["season"]==s, col].dropna() for s in seasons_present]
    bp = axes[i].boxplot(
        season_data, labels=seasons_present, patch_artist=True,
        medianprops=dict(color="black", lw=2),
    )
    for patch, clr in zip(bp["boxes"], palette_s):
        patch.set_facecolor(clr)
        patch.set_alpha(0.8)
    lbl = removal_labels[i]
    axes[i].set_title(f"{lbl} Removal by Season", fontsize=11, fontweight="bold")
    axes[i].set_ylabel("Removal (%)" if "log" not in col else "Log Removal", fontsize=9)
    axes[i].set_xlabel("Season", fontsize=9)
    axes[i].grid(True, alpha=0.3, axis="y")

fig.suptitle("Seasonal Variation in Treatment Performance", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / "04_seasonal_analysis.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  ✓ Saved 04_seasonal_analysis.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 5 ▸ WETLAND TYPE COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
print("[5/7] Wetland Type Comparison …")
wt_order = df["wetland_type"].value_counts().index.tolist()

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()
palette_w = sns.color_palette("tab10", len(wt_order))

for i, col in enumerate(REMOVAL_COLS):
    if col not in df.columns:
        continue
    means  = df.groupby("wetland_type")[col].mean().reindex(wt_order)
    stds   = df.groupby("wetland_type")[col].std().reindex(wt_order)
    bars = axes[i].bar(
        range(len(wt_order)), means.values,
        yerr=stds.values, capsize=4,
        color=palette_w, edgecolor="white", alpha=0.85,
    )
    axes[i].set_xticks(range(len(wt_order)))
    axes[i].set_xticklabels(
        [w.replace(" ", "\n") for w in wt_order],
        fontsize=8,
    )
    axes[i].set_title(f"{removal_labels[i]} Removal by Wetland Type", fontsize=10, fontweight="bold")
    axes[i].set_ylabel("Mean Removal (%)" if "log" not in col else "Mean Log Removal", fontsize=9)
    axes[i].grid(True, alpha=0.3, axis="y")

fig.suptitle("Treatment Performance by Wetland Configuration", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / "05_wetland_type_comparison.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  ✓ Saved 05_wetland_type_comparison.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 6 ▸ INFLUENT vs EFFLUENT
# ═══════════════════════════════════════════════════════════════════════════════
print("[6/7] Influent vs Effluent Comparison …")
pairs = list(zip(INFLUENT_PARAMS, EFFLUENT_PARAMS))
param_labels = ["BOD (mg/L)", "COD (mg/L)", "TSS (mg/L)", "TN (mg/L)", "TP (mg/L)", "pH"]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, ((inf_col, eff_col), lbl) in enumerate(zip(pairs, param_labels)):
    if inf_col not in df.columns or eff_col not in df.columns:
        continue
    inf_data = df[inf_col].dropna()
    eff_data = df[eff_col].dropna()
    axes[i].hist(inf_data, bins=40, alpha=0.65, color="#E74C3C", label=f"Influent (μ={inf_data.mean():.1f})", density=True)
    axes[i].hist(eff_data, bins=40, alpha=0.65, color="#27AE60", label=f"Effluent (μ={eff_data.mean():.1f})", density=True)
    axes[i].set_title(f"{lbl}", fontsize=11, fontweight="bold")
    axes[i].set_xlabel(lbl, fontsize=9)
    axes[i].set_ylabel("Density", fontsize=9)
    axes[i].legend(fontsize=8)
    axes[i].grid(True, alpha=0.3)

fig.suptitle("Influent vs Effluent Parameter Distributions", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / "06_influent_vs_effluent.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  ✓ Saved 06_influent_vs_effluent.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 7 ▸ FEATURE IMPORTANCE (RandomForest)
# ═══════════════════════════════════════════════════════════════════════════════
print("[7/7] Feature Importance …")

feat_cols = (
    DESIGN_PARAMS +
    ["water_temperature_C", "rainfall_mm_day",
     "mixed_source_domestic_pct", "mixed_source_industrial_pct",
     "mixed_source_agricultural_pct"] +
    INFLUENT_PARAMS
)
feat_cols = [c for c in feat_cols if c in df.columns]
X = df[feat_cols].copy()
y = df[TARGET].copy()
valid = y.notna() & X.notna().all(axis=1)
X, y = X[valid], y[valid]

imp = SimpleImputer(strategy="median")
X_imp = imp.fit_transform(X)

rf = RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42)
rf.fit(X_imp, y)
importances = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 9))
colors_fi = plt.cm.viridis(np.linspace(0.2, 0.9, len(importances)))
bars = ax.barh(importances.index, importances.values, color=colors_fi, edgecolor="white")
for bar, val in zip(bars, importances.values):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", ha="left", fontsize=8)
ax.set_xlabel("Feature Importance (Mean Decrease Impurity)", fontsize=11)
ax.set_title(f"Feature Importance for '{TARGET}'\n(RandomForest, n_estimators=200)", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3, axis="x")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUT_DIR / "07_feature_importance.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  ✓ Saved 07_feature_importance.png")

# ═══════════════════════════════════════════════════════════════════════════════
# BONUS ▸ OVERALL EFFICIENCY SCATTER MATRIX (top 6 features)
# ═══════════════════════════════════════════════════════════════════════════════
print("[Bonus] Scatter: Top features vs Overall Efficiency …")
top6 = importances.tail(6).index.tolist()
plot_df = df[top6 + [TARGET, "wetland_type"]].dropna()

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()
wt_types = plot_df["wetland_type"].unique()
cmap_wt  = dict(zip(wt_types, sns.color_palette("tab10", len(wt_types))))

for i, feat in enumerate(top6):
    for wt in wt_types:
        sub = plot_df[plot_df["wetland_type"] == wt]
        axes[i].scatter(
            sub[feat], sub[TARGET],
            c=[cmap_wt[wt]], alpha=0.3, s=8, label=wt,
        )
    # regression line
    m, b = np.polyfit(plot_df[feat], plot_df[TARGET], 1)
    xr = np.linspace(plot_df[feat].min(), plot_df[feat].max(), 200)
    axes[i].plot(xr, m*xr + b, color="black", lw=1.5, ls="--")
    axes[i].set_xlabel(feat.replace("_", " "), fontsize=9)
    axes[i].set_ylabel("Overall Efficiency (%)", fontsize=9)
    axes[i].set_title(feat.replace("_", " "), fontsize=10, fontweight="bold")
    axes[i].grid(True, alpha=0.3)

# shared legend
handles = [plt.Line2D([0],[0], marker="o", color="w",
            markerfacecolor=cmap_wt[wt], markersize=7, label=wt)
           for wt in wt_types]
fig.legend(handles=handles, title="Wetland Type",
           loc="lower center", ncol=len(wt_types), fontsize=8,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Top 6 Features vs Overall Treatment Efficiency", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / "08_top_features_scatter.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  ✓ Saved 08_top_features_scatter.png")

print("\n✅  All plots saved to:", OUT_DIR)