import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor,
)
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import os
import time

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="HCW AI — Constructed Wetland Efficiency",
    page_icon="🌿"
)

st.markdown("""
    <style>
    .main { background-color: #f4f7fb; }
    [data-testid="stSidebar"] { background-color: #0f1f3d !important; }
    [data-testid="stSidebar"] * { color: #ccd9f0 !important; }
    [data-testid="stSidebar"] .stRadio label { font-size: 1rem !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #90b8f8 !important; }
    [data-testid="metric-container"] {
        background: #ffffff;
        padding: 16px 18px;
        border-radius: 12px;
        border-left: 5px solid #16a34a;
        box-shadow: 0 2px 8px rgba(22,163,74,0.08);
    }
    [data-testid="stMetricLabel"]  { font-size: 0.80rem !important; color: #64748b !important; }
    [data-testid="stMetricValue"]  { font-size: 1.45rem !important; font-weight: 700 !important; color: #1e3a5f !important; }
    .section-header {
        background: linear-gradient(90deg, #0f1f3d, #15803d);
        color: #bbf7d0;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 0.97rem;
        font-weight: 700;
        letter-spacing: 0.4px;
        margin: 24px 0 12px 0;
        border-left: 5px solid #4ade80;
    }
    .status-alert {
        padding: 18px; border-radius: 12px; text-align: center;
        font-size: 1.2rem; font-weight: bold; margin-bottom: 14px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.09);
    }
    .pred-card {
        padding: 20px 22px; border-radius: 12px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.07);
        margin-bottom: 10px;
        text-align: center;
        background: #ffffff;
    }
    .pred-value { font-size: 2.2rem; font-weight: 800; color: #1e3a5f; }
    .pred-label { font-size: 0.88rem; color: #64748b; margin-top: 4px; }
    .pred-sub   { font-size: 0.82rem; color: #94a3b8; margin-top: 3px; }
    .info-box {
        background: #f0fdf4; border-radius: 10px; padding: 12px 16px;
        border-left: 5px solid #16a34a; margin: 10px 0; font-size: 0.92rem;
        color: #1e3a5f;
    }
    h1 { color: #0f1f3d !important; }
    h2 { color: #1d3461 !important; }
    h3 { color: #15803d !important; }
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #0f1f3d, #15803d) !important;
        color: white !important; font-weight: 700 !important;
        border-radius: 10px !important; font-size: 1rem !important;
        padding: 0.55rem 1.4rem !important;
        border: none !important;
    }
    hr { border-color: #bbf7d0 !important; }
    </style>
""", unsafe_allow_html=True)

if not os.path.exists("graphs"):
    os.makedirs("graphs")

DATA_PATH = "pune_hybrid_constructed_wetland_ml_dataset_28451_v2.xlsx"

# colour palette — green / teal / navy
ACCENT  = '#16a34a'
ACCENT2 = '#0891b2'
ACCENT3 = '#7c3aed'
WARN    = '#f59e0b'
DANGER  = '#ef4444'
NAVY    = '#0f1f3d'

TARGET = "overall_treatment_efficiency_pct"

# ─────────────────────────────────────────────────────────
# FEATURE CONFIG  (10 user-facing inputs)
# ─────────────────────────────────────────────────────────
MODEL_FEATURES = [
    'influent_BOD_mg_L', 'influent_COD_mg_L', 'influent_TSS_mg_L',
    'flow_m3_day', 'hydraulic_retention_time_day',
    'bed_area_m2', 'bed_depth_m',
    'water_temperature_C', 'influent_pH', 'plant_density_plants_m2',
]

FEATURE_META = {
    'influent_BOD_mg_L':              {'min': 50.0,  'max': 200.0, 'mean': 125.0, 'step': 5.0,  'unit': 'mg/L', 'label': 'Influent BOD'},
    'influent_COD_mg_L':              {'min': 80.0,  'max': 400.0, 'mean': 240.0, 'step': 10.0, 'unit': 'mg/L', 'label': 'Influent COD'},
    'influent_TSS_mg_L':              {'min': 50.0,  'max': 200.0, 'mean': 125.0, 'step': 5.0,  'unit': 'mg/L', 'label': 'Influent TSS'},
    'flow_m3_day':                    {'min': 10.0,  'max': 100.0, 'mean': 55.0,  'step': 1.0,  'unit': 'm³/day','label': 'Flow Rate'},
    'hydraulic_retention_time_day':   {'min': 10.0,  'max': 30.0,  'mean': 20.0,  'step': 0.5,  'unit': 'days', 'label': 'Hydraulic Retention Time'},
    'bed_area_m2':                    {'min': 1000.0,'max': 5000.0,'mean': 3000.0,'step': 10.0, 'unit': 'm²',   'label': 'Bed Area'},
    'bed_depth_m':                    {'min': 0.1,   'max': 2.0,   'mean': 1.05,  'step': 0.05, 'unit': 'm',    'label': 'Bed Depth'},
    'water_temperature_C':            {'min': 5.0,   'max': 50.0,  'mean': 27.5,  'step': 0.5,  'unit': '°C',   'label': 'Water Temperature'},
    'influent_pH':                    {'min': 5.0,   'max': 10.0,  'mean': 7.5,   'step': 0.1,  'unit': '',     'label': 'Influent pH'},
    'plant_density_plants_m2':        {'min': 1.0,   'max': 50.0,  'mean': 25.0,  'step': 0.5,  'unit': 'pl/m²','label': 'Plant Density'},
}

SCENARIO_PRESETS = {
    'Scenario A — High Performance': [80.0, 200.0, 100.0, 20.0, 15.0, 1000.0, 0.6, 28.0, 7.2, 15.0],
    'Scenario B — Moderate Load':    [180.0,400.0, 200.0, 50.0, 10.0,  500.0, 0.6, 25.0, 7.0, 10.0],
    'Scenario C — High Stress':      [450.0,1000.0,600.0,200.0,  3.0,  100.0, 1.5, 18.0, 5.5,  3.0],
}

# ─────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────
@st.cache_data
def load_dataset():
    if DATA_PATH.endswith(".xlsx"):
        df = pd.read_excel(DATA_PATH)
    else:
        df = pd.read_csv(DATA_PATH)
    # strip string columns
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()
    return df

# ─────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────
@st.cache_resource
def train_system(_df):
    """
    Trains 4 ML models using exact same logic as train_hcw_models.py:
      1. Hist Gradient Boosting  (lr=0.08, max_iter=250, max_leaf_nodes=31)
      2. Extra Trees             (n=180, min_samples_leaf=2)
      3. Random Forest           (n=160, max_depth=18, min_samples_leaf=2)
      4. Fast SVM Kernel Ridge   (Nystroem rbf + Ridge)
    Best model is used directly for prediction (full feature pipeline).
    """
    t0_total = time.time()
    RANDOM_STATE = 42
    df = _df.copy().dropna(subset=[TARGET])

    # ── 1. Leakage columns (same as train_hcw_models.leakage_columns) ──
    exact_drop = {
        "sample_id", "sample_date", "data_split",
        "CPCB_reuse_class", "maintenance_status", TARGET,
    }
    effluent_cols  = [c for c in df.columns if str(c).startswith("effluent_")]
    removal_cols_l = [c for c in df.columns
                      if str(c).endswith("_removal_pct") or c == "fecal_coliform_log_removal"]
    drop_cols = sorted(exact_drop | set(effluent_cols) | set(removal_cols_l))
    drop_cols = [c for c in drop_cols if c in df.columns]

    feature_cols_all = [c for c in df.columns if c not in drop_cols]
    X_all = df[feature_cols_all].copy()
    y_all = df[TARGET].astype(float)

    # ── 2. Train / Val / Test split ──
    if "data_split" in df.columns:
        split = df["data_split"].astype(str).str.lower().str.strip()
        X_tr, y_tr = X_all[split == "train"],      y_all[split == "train"]
        X_va, y_va = X_all[split == "validation"], y_all[split == "validation"]
        X_te, y_te = X_all[split == "test"],       y_all[split == "test"]
    else:
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(X_all, y_all, test_size=0.30, random_state=RANDOM_STATE)
        X_va, X_te,  y_va, y_te  = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=RANDOM_STATE)

    # ── 3. Preprocessor (same as train_hcw_models.make_preprocessor) ──
    num_cols = X_tr.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = [c for c in X_tr.columns if c not in num_cols]

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot",  ohe),
        ]), cat_cols),
    ], remainder="drop")

    # ── 4. Models — exact hyperparameters from train_hcw_models.build_models ──
    models_def = {
        "Hist Gradient Boosting": Pipeline([
            ("preprocess", preprocessor),
            ("model", HistGradientBoostingRegressor(
                learning_rate=0.08, max_iter=250, max_leaf_nodes=31,
                l2_regularization=0.05, random_state=RANDOM_STATE)),
        ]),
        "Extra Trees": Pipeline([
            ("preprocess", preprocessor),
            ("model", ExtraTreesRegressor(
                n_estimators=180, min_samples_leaf=2,
                random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "Random Forest": Pipeline([
            ("preprocess", preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=160, max_depth=18, min_samples_leaf=2,
                random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "Fast SVM Kernel Ridge": Pipeline([
            ("preprocess", preprocessor),
            ("kernel", Nystroem(kernel="rbf", gamma=0.04, n_components=450,
                                random_state=RANDOM_STATE)),
            ("model", Ridge(alpha=1.0)),
        ]),
    }

    # ── 5. Metrics (same as train_hcw_models.regression_metrics) ──
    def reg_metrics(yt, yp):
        yt, yp = np.asarray(yt, dtype=float), np.asarray(yp, dtype=float)
        r2   = float(r2_score(yt, yp))
        mae  = float(mean_absolute_error(yt, yp))
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        mask = yt != 0
        mape = float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100) if mask.any() else float("nan")
        acc  = round(100 - mape, 2) if not np.isnan(mape) else float("nan")
        return {
            "r2": round(r2, 4), "r2_pct": round(r2 * 100, 2),
            "mae": round(mae, 4), "rmse": round(rmse, 4),
            "mape": round(mape, 4) if not np.isnan(mape) else mape,
            "accuracy": acc,
        }

    # ── 6. Train all models ──
    results, trained, model_times = [], {}, {}
    for mname, pipeline in models_def.items():
        t0 = time.time()
        pipeline.fit(X_tr, y_tr)
        elapsed = round(time.time() - t0, 1)
        model_times[mname] = elapsed

        va_m = reg_metrics(y_va.values, pipeline.predict(X_va))
        te_m = reg_metrics(y_te.values, pipeline.predict(X_te))

        results.append({"model": mname, "train_s": elapsed,
                        **{f"val_{k}": v for k, v in va_m.items()},
                        **{f"test_{k}": v for k, v in te_m.items()}})
        trained[mname] = pipeline

    results_df = pd.DataFrame(results).sort_values("test_r2", ascending=False)
    best_name    = str(results_df.iloc[0]["model"])
    best_pipeline = trained[best_name]

    # ── 7. Feature importance (only tree-based models have it) ──
    best_model_obj = best_pipeline.named_steps["model"]
    feat_imp = None
    if hasattr(best_model_obj, "feature_importances_"):
        try:
            feat_names = list(best_pipeline.named_steps["preprocess"].get_feature_names_out())
        except Exception:
            feat_names = num_cols + cat_cols
        n = len(best_model_obj.feature_importances_)
        feat_imp = pd.Series(
            best_model_obj.feature_importances_,
            index=feat_names[:n]
        ).sort_values(ascending=False).head(20)

    # ── 8. Removal regressors for BOD / COD / TSS ──
    removal_target_cols = ["BOD_removal_pct", "COD_removal_pct", "TSS_removal_pct"]
    removal_regs, removal_metrics = {}, {}
    for rc in removal_target_cols:
        if rc in df.columns:
            y_rc  = df[rc].astype(float)
            valid = y_rc.notna() & X_all.notna().all(axis=1)
            Xv, yv = X_all[valid], y_rc[valid]
            Xv_tr, Xv_te, yv_tr, yv_te = train_test_split(Xv, yv, test_size=0.2, random_state=RANDOM_STATE)
            rr = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=RANDOM_STATE)
            pp = Pipeline([("preprocess", preprocessor), ("model", rr)])
            pp.fit(Xv_tr, yv_tr)
            removal_regs[rc]    = pp
            removal_metrics[rc] = reg_metrics(yv_te.values, pp.predict(Xv_te))

    # ── 9. Manual prediction: use best full pipeline directly ──
    #    Fill all feature columns with median (numeric) or mode (categorical)
    #    so best_pipeline can predict from only the 10 user-facing MODEL_FEATURES.
    median_fill = {}
    for col in feature_cols_all:
        col_data = X_tr[col].dropna()
        if col_data.empty:
            median_fill[col] = 0
        elif X_tr[col].dtype.kind in "biufc":   # numeric
            median_fill[col] = float(col_data.median())
        else:                                    # categorical
            median_fill[col] = col_data.mode()[0]

    return {
        "trained":         trained,
        "best_name":       best_name,
        "best_pipeline":   best_pipeline,
        "results_df":      results_df,
        "model_times":     model_times,
        "feat_imp":        feat_imp,
        "removal_regs":    removal_regs,
        "removal_metrics": removal_metrics,
        "median_fill":     median_fill,
        "feature_cols":    feature_cols_all,
        "total":           len(df),
        "df":              df,
        "train_time":      round(time.time() - t0_total, 1),
        "test_metrics":    {mname: {k: v for k, v in row.items() if k.startswith("test_")}
                            for mname, row in zip(
                                results_df["model"], results_df.to_dict("records"))},
    }


# ─────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────
for key in ["results", "df"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
st.sidebar.title("🌿 HCW AI")
st.sidebar.caption("Hybrid Constructed Wetland — Efficiency Intelligence")
st.sidebar.markdown("---")
page = st.sidebar.radio("📌 Navigate to:", [
    "🏠 Research Dashboard",
    "🔮 Manual Prediction",
    "📊 Statistical Visualization",
])


# ══════════════════════════════════════════════════════════
# PAGE 1 — RESEARCH DASHBOARD
# ══════════════════════════════════════════════════════════
if page == "🏠 Research Dashboard":
    st.title("🌿 HCW AI — Hybrid Constructed Wetland Efficiency Intelligence System")
    st.markdown(
        "*Predicts Overall Treatment Efficiency and pollutant removal rates from wetland "
        "design and influent data using **Hist Gradient Boosting**, **Extra Trees**, and **Random Forest**.*"
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 Run Full System Audit", type="primary"):
        with st.spinner("Loading dataset & training models…"):
            try:
                df = load_dataset()
                st.session_state.df = df
                st.session_state.results = train_system(df)
                st.success(
                    f"All models trained in {st.session_state.results['train_time']}s on "
                    f"{st.session_state.results['total']:,} records."
                )
            except FileNotFoundError:
                st.error(
                    f"Dataset not found: `{DATA_PATH}`\n\n"
                    "Please place `pune_hybrid_constructed_wetland_ml_dataset_28451_v2.xlsx` "
                    "in the same folder as this app."
                )

    if st.session_state.results:
        res = st.session_state.results
        rdf = res["results_df"]
        best = res["best_name"]

        best_r2  = rdf.iloc[0]["test_r2_pct"]
        best_acc = rdf.iloc[0]["test_accuracy"]
        best_mae = rdf.iloc[0]["test_mae"]
        best_rmse= rdf.iloc[0]["test_rmse"]

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Total Records",   f"{res['total']:,}")
        k2.metric("Best Model",       best)
        k3.metric("Pred. Accuracy",  f"{best_acc:.2f}%")
        k4.metric("R² Score",        f"{best_r2:.2f}%")
        k5.metric("Best MAE",        f"{best_mae:.3f}")
        k6.metric("Best RMSE",       f"{best_rmse:.3f}")

        st.markdown(f"""
        <div class="info-box">
        <b>Target:</b> Overall Treatment Efficiency (%) &nbsp;|&nbsp;
        <b>Best Model:</b> {best} &nbsp;|&nbsp;
        <b>Prediction Accuracy:</b> {best_acc:.2f}% &nbsp;|&nbsp;
        <b>R²:</b> {best_r2:.2f}%
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.subheader("📊 Model Accuracy Comparison")
            model_colors = [ACCENT, ACCENT2, ACCENT3]
            fig_bar = go.Figure()
            for (_, row), mc in zip(rdf.iterrows(), model_colors):
                fig_bar.add_trace(go.Bar(
                    x=[row["model"]], y=[row["test_accuracy"]], marker_color=mc,
                    text=[f"{row['test_accuracy']:.2f}%"], textposition="outside",
                    name=row["model"], width=0.45,
                    customdata=[res["model_times"].get(row["model"], "-")],
                    hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.2f}%<br>Train Time: %{customdata}s<extra></extra>"
                ))
            fig_bar.update_layout(
                yaxis=dict(range=[50, 105], title="Prediction Accuracy (%)"),
                showlegend=False, template="plotly_white",
                title="3 ML Models — Prediction Accuracy (100 − MAPE)",
                height=320, margin=dict(t=50, b=10)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_b:
            st.subheader("📐 R² by Model")
            fig_r2 = px.bar(
                rdf, x="model", y="test_r2_pct",
                color="model", text="test_r2_pct",
                color_discrete_sequence=[ACCENT, ACCENT2, ACCENT3],
                template="plotly_white",
                title="R² Score (%)"
            )
            fig_r2.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            fig_r2.update_layout(showlegend=False, height=320,
                                 yaxis=dict(range=[0, 105]),
                                 margin=dict(t=50, b=10))
            st.plotly_chart(fig_r2, use_container_width=True)

        st.divider()

        # Regression metrics table
        st.subheader("📋 Full Model Performance Summary")
        rows = []
        for _, row in rdf.iterrows():
            rows.append({
                "Model":            row["model"],
                "Accuracy (%)":     f"{row['test_accuracy']:.2f}",
                "R² (%)":           f"{row['test_r2_pct']:.2f}",
                "MAE":              f"{row['test_mae']:.3f}",
                "RMSE":             f"{row['test_rmse']:.3f}",
                "MAPE (%)":         f"{row['test_mape']:.2f}",
                "Train Time (s)":   res["model_times"].get(row["model"], "-"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.divider()

        # Feature importance
        if res["feat_imp"] is not None:
            st.subheader("🔑 Feature Importance — Best Model")
            fi = res["feat_imp"].head(15)
            fig_fi = px.bar(
                x=fi.values, y=fi.index,
                orientation="h",
                labels={"x": "Importance", "y": "Feature"},
                color=fi.values, color_continuous_scale="Greens",
                title=f"Top 15 Predictors of Treatment Efficiency ({best})",
                template="plotly_white"
            )
            fig_fi.update_layout(coloraxis_showscale=False, height=420,
                                 margin=dict(t=50), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_fi, use_container_width=True)

        with st.expander("📖 About This System"):
            st.markdown(f"""
            **Project:** *Machine Learning Evaluation and Design Insights of Hybrid Constructed
            Wetlands for Multi-Source Wastewater Treatment*  
            Anantrao Pawar College of Engineering & Research, Savitribai Phule Pune University, A.Y. 2025-26

            | Model | Role | Accuracy / R² |
            |---|---|---|
            | Hist Gradient Boosting | Primary regressor | {rdf[rdf['model']=='Hist Gradient Boosting']['test_accuracy'].values[0] if 'Hist Gradient Boosting' in rdf['model'].values else 'N/A'}% |
            | Extra Trees | Ensemble regressor | {rdf[rdf['model']=='Extra Trees']['test_accuracy'].values[0] if 'Extra Trees' in rdf['model'].values else 'N/A'}% |
            | Random Forest | Baseline ensemble | {rdf[rdf['model']=='Random Forest']['test_accuracy'].values[0] if 'Random Forest' in rdf['model'].values else 'N/A'}% |
            """)
    else:
        st.info("Click **Run Full System Audit** in the sidebar to load data and train all models.")


# ══════════════════════════════════════════════════════════
# PAGE 2 — MANUAL PREDICTION
# ══════════════════════════════════════════════════════════
elif page == "🔮 Manual Prediction":
    st.title("🔮 HCW Prediction Lab")
    st.markdown(
        "Enter wetland design and influent parameters to receive a **data-driven prediction** "
        "of Overall Treatment Efficiency, along with engineering recommendations."
    )

    if not st.session_state.results:
        st.warning("Please run **System Audit** on the Research Dashboard first.")
        st.stop()

    res = st.session_state.results

    # default values = dataset means
    pv = [FEATURE_META[f]["mean"] for f in MODEL_FEATURES]

    st.markdown("### Step 1 — Enter Parameters")

    st.markdown("#### 🧪 Influent Quality")
    i1, i2, i3 = st.columns(3)
    with i1:
        inf_bod = st.number_input(
            "Influent BOD (mg/L)", min_value=50.0, max_value=200.0,
            value=float(np.clip(pv[0], 50, 200)), step=5.0, format="%.1f",
            help="Valid range: 50 - 200 mg/L")
    with i2:
        inf_cod = st.number_input(
            "Influent COD (mg/L)", min_value=80.0, max_value=400.0,
            value=float(np.clip(pv[1], 80, 400)), step=10.0, format="%.1f",
            help="Valid range: 80 - 400 mg/L")
    with i3:
        inf_tss = st.number_input(
            "Influent TSS (mg/L)", min_value=50.0, max_value=200.0,
            value=float(np.clip(pv[2], 50, 200)), step=5.0, format="%.1f",
            help="Valid range: 50 - 200 mg/L")

    st.markdown("#### 💧 Hydraulic Parameters")
    h1, h2, h3 = st.columns(3)
    with h1:
        flow = st.number_input(
            "Flow Rate (m³/day)", min_value=10.0, max_value=100.0,
            value=float(np.clip(pv[3], 10, 100)), step=1.0, format="%.1f",
            help="Valid range: 10 - 100 m3/day")
    with h2:
        hrt = st.number_input(
            "Hydraulic Retention Time (days)", min_value=10.0, max_value=30.0,
            value=float(np.clip(pv[4], 10, 30)), step=0.5, format="%.2f",
            help="Valid range: 10 - 30 days")
    with h3:
        bed_area = st.number_input(
            "Bed Area (m²)", min_value=1000.0, max_value=5000.0,
            value=float(np.clip(pv[5], 1000, 5000)), step=10.0, format="%.1f",
            help="Valid range: 1000 - 5000 m2")

    st.markdown("#### 🏗️ Design & Environmental")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        bed_depth = st.number_input(
            "Bed Depth (m)", min_value=0.1, max_value=2.0,
            value=float(np.clip(pv[6], 0.1, 2.0)), step=0.05, format="%.2f",
            help="Valid range: 0.1 - 2.0 m")
    with d2:
        temp = st.number_input(
            "Water Temperature (°C)", min_value=5.0, max_value=50.0,
            value=float(np.clip(pv[7], 5, 50)), step=0.5, format="%.1f",
            help="Valid range: 5 - 50 C")
    with d3:
        ph = st.slider(
            "Influent pH", min_value=5.0, max_value=10.0,
            value=float(np.clip(pv[8], 5.0, 10.0)), step=0.05,
            help="Valid range: 5.0 - 10.0")
    with d4:
        plant_density = st.number_input(
            "Plant Density (plants/m²)", min_value=1.0, max_value=50.0,
            value=float(np.clip(pv[9], 1.0, 50.0)), step=0.5, format="%.1f",
            help="Valid range: 1 - 50 plants/m2")

    user_input = [inf_bod, inf_cod, inf_tss, flow, hrt, bed_area, bed_depth, temp, ph, plant_density]

    st.divider()
    predict_clicked = st.button("🌿 Run Prediction", type="primary", use_container_width=True)

    if predict_clicked:
        with st.spinner("Running prediction..."):
            # Build full-feature row: median fill for all columns, override with user inputs
            row_dict = dict(res["median_fill"])
            for feat, val in zip(MODEL_FEATURES, user_input):
                row_dict[feat] = val
            X_pred = pd.DataFrame([row_dict])[res["feature_cols"]]
            prediction = float(res["best_pipeline"].predict(X_pred)[0])
            prediction = round(np.clip(prediction, 0, 100), 2)

        best_model_name = res["best_name"]
        st.markdown("### Step 2 — Prediction Results")

        # System classification
        if prediction >= 85:
            classification = "Excellent Performance"
            cls_color = ACCENT
        elif prediction >= 70:
            classification = "Acceptable Performance"
            cls_color = WARN
        else:
            classification = "Performance Improvement Required"
            cls_color = DANGER

        osi = round(prediction / 100, 2)

        st.markdown(
            f'<div class="status-alert" style="background:#f0fdf4;color:{NAVY};border:2px solid {cls_color};">'
            f'Overall Treatment Efficiency: <b>{prediction:.2f}%</b>'
            f'  |  Classification: <b>{classification}</b>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Three prediction cards
        st.markdown("#### Key Prediction Metrics")
        p1, p2, p3 = st.columns(3)

        with p1:
            st.markdown(f"""
            <div class="pred-card" style="border-top:4px solid {ACCENT};">
                <div class="pred-value">{prediction:.2f}<span style="font-size:1rem;font-weight:400;color:#94a3b8;"> %</span></div>
                <div class="pred-label">Overall Treatment Efficiency</div>
                <div class="pred-sub">Predicted by {best_model_name}</div>
            </div>""", unsafe_allow_html=True)

        with p2:
            st.markdown(f"""
            <div class="pred-card" style="border-top:4px solid {ACCENT2};">
                <div class="pred-value">{osi}</div>
                <div class="pred-label">Operational Suitability Index</div>
                <div class="pred-sub">Scale: 0.0 (poor) → 1.0 (excellent)</div>
            </div>""", unsafe_allow_html=True)

        with p3:
            st.markdown(f"""
            <div class="pred-card" style="border-top:4px solid {cls_color};">
                <div class="pred-value" style="font-size:1.3rem;">{classification}</div>
                <div class="pred-label">System Classification</div>
                <div class="pred-sub">Based on efficiency threshold</div>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # Gauge
        st.subheader("Efficiency Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Overall Treatment Efficiency (%)", "font": {"size": 16}},
            number={"font": {"size": 42, "color": NAVY}, "suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": cls_color},
                "bgcolor": "#f1f5f9",
                "bordercolor": "#e2e8f0",
                "steps": [
                    {"range": [0, 70],  "color": "#fef9c3"},
                    {"range": [70, 85], "color": "#dcfce7"},
                    {"range": [85, 100],"color": "#bbf7d0"},
                ],
                "threshold": {
                    "line": {"color": DANGER, "width": 3},
                    "thickness": 0.75, "value": 70
                }
            }
        ))
        fig_gauge.update_layout(height=320, margin=dict(t=30, b=10, l=20, r=20),
                                template="plotly_white")
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.divider()

        # Parameter radar + assessment
        col_rd, col_rep = st.columns(2)

        with col_rd:
            st.subheader("Parameter Profile Radar")
            norm_vals = []
            for feat, val in zip(MODEL_FEATURES, user_input):
                mn = FEATURE_META[feat]["min"]
                mx = FEATURE_META[feat]["max"]
                norm_vals.append((val - mn) / (mx - mn + 1e-9))
            labels = [FEATURE_META[f]["label"] for f in MODEL_FEATURES]
            fig_radar = go.Figure(go.Scatterpolar(
                r=norm_vals + [norm_vals[0]],
                theta=labels + [labels[0]],
                fill="toself", line_color=ACCENT, fillcolor=ACCENT,
                opacity=0.20, name="Input"
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                template="plotly_white", height=380,
                title="Normalized Input (0 = min, 1 = max)"
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_rep:
            st.subheader("Engineering Assessment Report")

            # Parameter assessments
            assessments = []
            if bed_area > 1000:
                assessments.append(("✓ Large Bed Area", ACCENT))
            else:
                assessments.append(("⚠ Limited Bed Area", WARN))
            if hrt >= 10:
                assessments.append(("✓ Adequate HRT", ACCENT))
            else:
                assessments.append(("⚠ Low Hydraulic Retention Time", WARN))
            if 6.5 <= ph <= 8.0:
                assessments.append(("✓ Optimal Influent pH", ACCENT))
            elif ph < 6:
                assessments.append(("⚠ Acidic Influent pH", DANGER))
            else:
                assessments.append(("⚠ Alkaline Influent pH", WARN))
            if bed_depth > 1.2:
                assessments.append(("⚠ Excessive Bed Depth", WARN))
            else:
                assessments.append(("✓ Practical Bed Depth", ACCENT))
            if plant_density >= 10:
                assessments.append(("✓ Adequate Plant Density", ACCENT))
            else:
                assessments.append(("⚠ Low Plant Density", WARN))

            # Recommendations
            recs = []
            if ph < 6 or ph > 8.5:
                recs.append("Maintain pH between 6.5 – 8.0 for optimum biological treatment.")
            if plant_density < 10:
                recs.append("Increase vegetation density to enhance pollutant removal.")
            if bed_depth > 1.2:
                recs.append("Reduce bed depth to practical design range (0.3 – 1.2 m).")
            if hrt < 5:
                recs.append("Increase hydraulic retention time for better treatment efficiency.")
            recs.append("Continue monitoring organic loading and hydraulic performance.")

            report_rows = [
                ("Predicted Efficiency",      f"{prediction:.2f}%"),
                ("Classification",            classification),
                ("OSI",                       str(osi)),
                ("Bed Area",                  f"{bed_area:.1f} m²"),
                ("HRT",                       f"{hrt:.1f} days"),
                ("Influent pH",               str(ph)),
                ("Bed Depth",                 f"{bed_depth:.2f} m"),
                ("Plant Density",             f"{plant_density:.1f} pl/m²"),
                ("Key Driver 1",              "Hydraulic Retention Time"),
                ("Key Driver 2",              "Influent BOD / COD"),
                ("Key Driver 3",              "Bed Area & Depth"),
            ]
            st.dataframe(
                pd.DataFrame(report_rows, columns=["Parameter", "Value"]),
                use_container_width=True, hide_index=True
            )

            st.markdown("**Parameter Assessment:**")
            for txt, _ in assessments:
                st.markdown(f"- {txt}")

            st.markdown("**Engineering Recommendations:**")
            for r in recs:
                st.markdown(f"• {r}")


# ══════════════════════════════════════════════════════════
# PAGE 3 — STATISTICAL VISUALIZATION
# ══════════════════════════════════════════════════════════
elif page == "📊 Statistical Visualization":
    st.title("📊 HCW AI — Statistical Analysis & Visualization")

    if not st.session_state.results:
        st.info("Run System Audit on the Research Dashboard first.")
        st.stop()

    res = st.session_state.results
    df  = res["df"]

    REMOVAL_COLS = ["BOD_removal_pct","COD_removal_pct","TSS_removal_pct",
                    "TN_removal_pct", "TP_removal_pct"]
    removal_labels = ["BOD Removal","COD Removal","TSS Removal","TN Removal","TP Removal"]
    available_removals = [c for c in REMOVAL_COLS if c in df.columns]

    # 1. Correlation Heatmap
    st.markdown('<div class="section-header">1 · Feature Correlation Heatmap</div>', unsafe_allow_html=True)
    corr_cols = [c for c in (
        ["hydraulic_retention_time_day","bed_area_m2","bed_depth_m",
         "flow_m3_day","water_temperature_C","influent_pH",
         "influent_BOD_mg_L","influent_COD_mg_L","influent_TSS_mg_L",
         TARGET]
    ) if c in df.columns]
    corr = df[corr_cols].corr()
    short = [c.replace("hydraulic_retention_time_day","HRT")
              .replace("influent_","Inf ")
              .replace("_mg_L"," mg/L")
              .replace("_m2"," m²")
              .replace("_m"," m")
              .replace("_C"," °C")
              .replace("water_temperature","Temp")
              .replace("overall_treatment_efficiency_pct","OTE %")
              .replace("flow_m3_day","Flow")
              .replace("bed_area","Bed Area")
              .replace("bed_depth","Depth")
              .replace("_"," ").title()
              for c in corr_cols]
    fig_heat = go.Figure(data=go.Heatmap(
        z=corr.values, x=short, y=short,
        colorscale="RdBu", text=np.round(corr.values, 2),
        texttemplate="%{text}", zmin=-1, zmax=1
    ))
    fig_heat.update_layout(title="Pearson Correlation — Key HCW Parameters",
                           height=500, template="plotly_white")
    st.plotly_chart(fig_heat, use_container_width=True)

    # 2. Removal Efficiency Distributions (violin)
    if available_removals:
        st.markdown('<div class="section-header">2 · Pollutant Removal Efficiency Distributions</div>', unsafe_allow_html=True)
        melt_df = df[available_removals].melt(var_name="Pollutant", value_name="Removal (%)")
        melt_df["Pollutant"] = melt_df["Pollutant"].str.replace("_removal_pct","").str.upper()
        fig_viol = px.violin(melt_df, x="Pollutant", y="Removal (%)",
                             color="Pollutant", box=True, points="outliers",
                             color_discrete_sequence=[ACCENT,ACCENT2,ACCENT3,WARN,DANGER],
                             title="Pollutant Removal Efficiency Distribution",
                             template="plotly_white")
        fig_viol.update_layout(showlegend=False, height=420)
        st.plotly_chart(fig_viol, use_container_width=True)

    # 3. Mean metrics by wetland type
    if "wetland_type" in df.columns:
        st.markdown('<div class="section-header">3 · Mean Treatment Performance by Wetland Type</div>', unsafe_allow_html=True)
        wt_cols = [TARGET] + [c for c in available_removals[:3]]
        wt_agg  = df.groupby("wetland_type")[wt_cols].mean().reset_index()
        fig_wt  = go.Figure()
        for col_n, col_c in zip(wt_cols, [ACCENT, ACCENT2, ACCENT3, WARN]):
            fig_wt.add_trace(go.Bar(name=col_n.replace("_"," ").replace("pct","%"),
                                     x=wt_agg["wetland_type"], y=wt_agg[col_n],
                                     marker_color=col_c,
                                     text=wt_agg[col_n].round(1), textposition="outside"))
        fig_wt.update_layout(barmode="group", template="plotly_white",
                             title="Mean Treatment Metrics by Wetland Type",
                             yaxis_title="Mean Value (%)", height=440,
                             legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_wt, use_container_width=True)

    # 4. Seasonal pattern
    if "season" in df.columns and TARGET in df.columns:
        st.markdown('<div class="section-header">4 · Seasonal Treatment Efficiency Pattern</div>', unsafe_allow_html=True)
        season_order = ["Winter","Summer","Monsoon","Post-monsoon"]
        seasons_ok   = [s for s in season_order if s in df["season"].unique()]
        means = df.groupby("season")[TARGET].mean().reindex(seasons_ok)
        stds  = df.groupby("season")[TARGET].std().reindex(seasons_ok)
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(
            x=means.index, y=means.values,
            mode="lines+markers", name="Mean Efficiency",
            line=dict(color=ACCENT, width=2.5),
            error_y=dict(type="data", array=stds.values, visible=True, color=ACCENT)
        ))
        if "Monsoon" in seasons_ok:
            fig_s.add_vrect(x0="Monsoon", x1="Monsoon",
                            fillcolor="#bbf7d0", opacity=0.3,
                            annotation_text="Monsoon", annotation_position="top left")
        fig_s.update_layout(template="plotly_white", height=400,
                            title="Seasonal Variation in Overall Treatment Efficiency",
                            yaxis_title="Efficiency (%)", xaxis_title="Season")
        st.plotly_chart(fig_s, use_container_width=True)

    # 5. HRT vs Efficiency scatter
    if "hydraulic_retention_time_day" in df.columns:
        st.markdown('<div class="section-header">5 · HRT vs Overall Treatment Efficiency</div>', unsafe_allow_html=True)
        sample5k = df.sample(min(3000, len(df)), random_state=42)
        color_col = "wetland_type" if "wetland_type" in df.columns else None
        fig_sc = px.scatter(sample5k,
                            x="hydraulic_retention_time_day", y=TARGET,
                            color=color_col,
                            color_discrete_sequence=[ACCENT, ACCENT2, ACCENT3, WARN, DANGER],
                            hover_data=["bed_area_m2","influent_BOD_mg_L"],
                            title="Hydraulic Retention Time vs Overall Treatment Efficiency",
                            opacity=0.55, template="plotly_white")
        fig_sc.update_layout(height=450,
                             xaxis_title="HRT (days)",
                             yaxis_title="Overall Treatment Efficiency (%)")
        st.plotly_chart(fig_sc, use_container_width=True)

    # 6. Feature Importance
    if res["feat_imp"] is not None:
        st.markdown('<div class="section-header">6 · Feature Importance — Best Model</div>', unsafe_allow_html=True)
        fi = res["feat_imp"].head(15)
        fig_fi = px.bar(
            x=fi.values, y=fi.index, orientation="h",
            labels={"x": "Importance", "y": "Feature"},
            color=fi.values, color_continuous_scale="Greens",
            title="Top 15 Important Features",
            template="plotly_white"
        )
        fig_fi.update_layout(coloraxis_showscale=False, height=460,
                             yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_fi, use_container_width=True)

    # 7. BOD vs Efficiency scatter
    if "influent_BOD_mg_L" in df.columns:
        st.markdown('<div class="section-header">7 · Influent BOD vs Overall Treatment Efficiency</div>', unsafe_allow_html=True)
        samp = df.sample(min(3000, len(df)), random_state=7)
        color_col2 = "wetland_type" if "wetland_type" in df.columns else None
        fig_bod = px.scatter(samp, x="influent_BOD_mg_L", y=TARGET,
                             color=color_col2,
                             color_discrete_sequence=[ACCENT, ACCENT2, ACCENT3, WARN, DANGER],
                             trendline="ols",
                             title="Influent BOD vs Overall Treatment Efficiency",
                             opacity=0.50, template="plotly_white",
                             labels={"influent_BOD_mg_L": "Influent BOD (mg/L)",
                                     TARGET: "Overall Treatment Efficiency (%)"})
        fig_bod.update_layout(height=430)
        st.plotly_chart(fig_bod, use_container_width=True)

    # 8. Actual vs Predicted (best model on test set)
    st.markdown('<div class="section-header">8 · Actual vs Predicted — Best Model</div>', unsafe_allow_html=True)
    best_pipe = res["best_pipeline"]
    feature_cols_avp = res["feature_cols"]
    df_clean = df.dropna(subset=[TARGET])
    split_col = df_clean["data_split"].astype(str).str.lower().str.strip() if "data_split" in df_clean.columns else None
    if split_col is not None:
        X_avp = df_clean[feature_cols_avp][split_col == "test"]
        y_avp = df_clean[TARGET][split_col == "test"]
    else:
        from sklearn.model_selection import train_test_split as _tts
        _, X_avp, _, y_avp = _tts(df_clean[feature_cols_avp], df_clean[TARGET], test_size=0.15, random_state=42)
    y_pred_avp = best_pipe.predict(X_avp)
    avp_df = pd.DataFrame({"Actual": y_avp.values, "Predicted": np.round(y_pred_avp, 2)})
    avp_df["Residual"] = np.round(avp_df["Actual"] - avp_df["Predicted"], 2)
    fig_avp = px.scatter(avp_df.sample(min(2000, len(avp_df)), random_state=1),
                         x="Actual", y="Predicted",
                         color="Residual", color_continuous_scale="RdYlGn",
                         title=f"Actual vs Predicted — {res['best_name']} (Test Set)",
                         opacity=0.6, template="plotly_white",
                         labels={"Actual": "Actual Efficiency (%)", "Predicted": "Predicted Efficiency (%)"})
    mn_val = float(min(avp_df["Actual"].min(), avp_df["Predicted"].min())) - 2
    mx_val = float(max(avp_df["Actual"].max(), avp_df["Predicted"].max())) + 2
    fig_avp.add_shape(type="line", x0=mn_val, y0=mn_val, x1=mx_val, y1=mx_val,
                      line=dict(color=DANGER, width=2, dash="dash"))
    fig_avp.update_layout(height=460)
    st.plotly_chart(fig_avp, use_container_width=True)

    # 9. Temperature vs Efficiency line
    if "water_temperature_C" in df.columns:
        st.markdown('<div class="section-header">9 · Water Temperature vs Treatment Efficiency</div>', unsafe_allow_html=True)
        temp_bins = pd.cut(df["water_temperature_C"], bins=15)
        temp_grp  = df.groupby(temp_bins, observed=True)[TARGET].agg(["mean","std","count"]).reset_index()
        temp_grp["mid"] = temp_grp["water_temperature_C"].apply(lambda x: round(x.mid, 1))
        temp_grp = temp_grp[temp_grp["count"] >= 10]
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(
            x=temp_grp["mid"], y=temp_grp["mean"],
            mode="lines+markers",
            line=dict(color=ACCENT2, width=2.5),
            marker=dict(size=7, color=ACCENT2),
            error_y=dict(type="data", array=temp_grp["std"].values, visible=True, color=ACCENT2),
            name="Mean ± SD"
        ))
        fig_temp.update_layout(template="plotly_white", height=400,
                               title="Water Temperature Bins vs Mean Treatment Efficiency",
                               xaxis_title="Water Temperature (°C)",
                               yaxis_title="Mean Treatment Efficiency (%)")
        st.plotly_chart(fig_temp, use_container_width=True)

    # 10. Flow Rate distribution with efficiency overlay
    if "flow_m3_day" in df.columns:
        st.markdown('<div class="section-header">10 · Flow Rate Distribution & Efficiency</div>', unsafe_allow_html=True)
        fc1, fc2 = st.columns(2)
        with fc1:
            fig_flow_hist = px.histogram(df, x="flow_m3_day", nbins=40,
                                         color_discrete_sequence=[ACCENT],
                                         title="Flow Rate Distribution (m³/day)",
                                         template="plotly_white",
                                         labels={"flow_m3_day": "Flow Rate (m³/day)"})
            fig_flow_hist.update_layout(height=360, showlegend=False)
            st.plotly_chart(fig_flow_hist, use_container_width=True)
        with fc2:
            flow_bins = pd.cut(df["flow_m3_day"], bins=12)
            flow_grp  = df.groupby(flow_bins, observed=True)[TARGET].mean().reset_index()
            flow_grp["mid"] = flow_grp["flow_m3_day"].apply(lambda x: round(x.mid, 1))
            fig_flow_eff = go.Figure(go.Bar(
                x=flow_grp["mid"].astype(str), y=flow_grp[TARGET].round(1),
                marker_color=ACCENT3,
                text=flow_grp[TARGET].round(1), textposition="outside"
            ))
            fig_flow_eff.update_layout(template="plotly_white", height=360,
                                       title="Mean Efficiency by Flow Rate Bin",
                                       xaxis_title="Flow Rate Bin midpoint (m³/day)",
                                       yaxis_title="Mean Efficiency (%)")
            st.plotly_chart(fig_flow_eff, use_container_width=True)

    # Statistical summary table
    st.markdown('<div class="section-header">Dataset Statistical Summary</div>', unsafe_allow_html=True)
    summary_cols = [c for c in [
        "influent_BOD_mg_L","influent_COD_mg_L","influent_TSS_mg_L",
        "hydraulic_retention_time_day","bed_area_m2","bed_depth_m",
        "water_temperature_C","influent_pH","plant_density_plants_m2",
        TARGET
    ] if c in df.columns]
    st.dataframe(df[summary_cols].describe().round(3), use_container_width=True)
