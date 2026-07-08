"""
Machine Learning Assisted Evaluation of Hybrid Constructed Wetlands
Treating Multi-Source Wastewater

Project:
    Machine Learning Evaluation and Design Insights of Hybrid Constructed
    Wetlands for Multi-Source Wastewater Treatment

What this file does:
    1. Loads your Excel dataset automatically from the project folder.
    2. Selects the correct sheet automatically.
    3. Removes leakage columns such as effluent values and already-calculated
       removal percentages from input features.
    4. Trains 4 fast machine-learning models.
    5. Compares model performance.
    6. Saves the best model, predictions, results CSV, and plots.

Run:
    python train_hcw_models.py

Optional:
    python train_hcw_models.py --target BOD_removal_pct --out results_bod
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_DATASET = ""
DEFAULT_SHEET = ""
DEFAULT_TARGET = "overall_treatment_efficiency_pct"
RANDOM_STATE = 42


def find_default_dataset() -> Path:
    """
    Finds the dataset automatically from the same folder as this script.

    Your current dataset file is:
        pune_hybrid_constructed_wetland_ml_dataset_28451_v2.xlsx

    This function accepts any file matching:
        pune_hybrid_constructed_wetland_ml_dataset_*.xlsx
    """
    script_dir = Path(__file__).resolve().parent
    candidates = sorted(script_dir.glob("pune_hybrid_constructed_wetland_ml_dataset_*.xlsx"))

    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        "Dataset not found.\n\n"
        "Fix:\n"
        "1. Put your Excel dataset in the same folder as train_hcw_models.py, or\n"
        "2. Run with --data \"C:\\path\\to\\your_dataset.xlsx\""
    )


def make_one_hot_encoder() -> OneHotEncoder:
    """
    Creates OneHotEncoder in a way that works with old and new scikit-learn.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_dataset(path: str, sheet_name: str) -> pd.DataFrame:
    """
    Loads Excel or CSV data.

    If no path is passed, it auto-finds the dataset in the project folder.
    If no sheet is passed, it uses the first Excel sheet.
    """
    dataset_path = Path(path) if path else find_default_dataset()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if dataset_path.suffix.lower() in {".xlsx", ".xls"}:
        excel = pd.ExcelFile(dataset_path)
        selected_sheet = sheet_name if sheet_name in excel.sheet_names else excel.sheet_names[0]
        print(f"Using dataset: {dataset_path}")
        print(f"Using sheet: {selected_sheet}")
        return pd.read_excel(dataset_path, sheet_name=selected_sheet)

    if dataset_path.suffix.lower() == ".csv":
        print(f"Using dataset: {dataset_path}")
        return pd.read_csv(dataset_path)

    raise ValueError("Use an Excel .xlsx/.xls file or CSV file.")


def leakage_columns(df: pd.DataFrame, target: str) -> list[str]:
    """
    Removes columns that reveal the answer.

    This is important for a correct ML project. If we include effluent columns
    or already-calculated removal columns, the model can get fake high accuracy.
    """
    exact_drop = {
        "sample_id",
        "sample_date",
        "data_split",
        "CPCB_reuse_class",
        "maintenance_status",
        target,
    }

    effluent_cols = [col for col in df.columns if str(col).startswith("effluent_")]

    removal_cols = [
        col
        for col in df.columns
        if str(col).endswith("_removal_pct") or col == "fecal_coliform_log_removal"
    ]

    drop_cols = sorted(exact_drop | set(effluent_cols) | set(removal_cols))
    return [col for col in drop_cols if col in df.columns]


def split_data(df: pd.DataFrame, target: str):
    """
    Splits data into train, validation, and test.

    If your dataset has data_split column, it uses that.
    Otherwise it creates a 70/15/15 split.
    """
    if target not in df.columns:
        available = "\n".join(map(str, df.columns))
        raise ValueError(
            f"Target column not found: {target}\n\nAvailable columns:\n{available}"
        )

    drop_cols = leakage_columns(df, target)
    feature_cols = [col for col in df.columns if col not in drop_cols]

    X = df[feature_cols].copy()
    y = df[target].astype(float).copy()

    if "data_split" in df.columns:
        split = df["data_split"].astype(str).str.lower().str.strip()

        X_train = X[split == "train"]
        y_train = y[split == "train"]

        X_valid = X[split == "validation"]
        y_valid = y[split == "validation"]

        X_test = X[split == "test"]
        y_test = y[split == "test"]
    else:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=RANDOM_STATE,
        )
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            random_state=RANDOM_STATE,
        )

    if len(X_train) == 0 or len(X_valid) == 0 or len(X_test) == 0:
        raise ValueError(
            "Train/validation/test split has an empty part. "
            "Check values in the data_split column."
        )

    return X_train, y_train, X_valid, y_valid, X_test, y_test, feature_cols, drop_cols


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Prepares numeric and categorical columns for machine learning.
    """
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )


def build_models(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    """
    Builds 4 ML models.

    These are selected to balance accuracy and speed.
    """
    return {
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=0.08,
                        max_iter=250,
                        max_leaf_nodes=31,
                        l2_regularization=0.05,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=180,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=160,
                        max_depth=18,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "fast_svm_kernel_ridge": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "kernel",
                    Nystroem(
                        kernel="rbf",
                        gamma=0.04,
                        n_components=450,
                        random_state=RANDOM_STATE,
                    ),
                ),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
    }


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """
    Calculates regression metrics.

    For project reporting:
        Accuracy percent = R2 x 100
    """
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)

    r2 = r2_score(y_true_array, y_pred_array)
    mae = mean_absolute_error(y_true_array, y_pred_array)
    rmse = np.sqrt(mean_squared_error(y_true_array, y_pred_array))

    nonzero_mask = y_true_array != 0
    if np.any(nonzero_mask):
        mape = (
            np.mean(
                np.abs(
                    (y_true_array[nonzero_mask] - y_pred_array[nonzero_mask])
                    / y_true_array[nonzero_mask]
                )
            )
            * 100
        )
    else:
        mape = np.nan

    prediction_accuracy = 100 - mape if not np.isnan(mape) else np.nan

    return {
        "r2": round(float(r2), 4),
        "r2_percent": round(float(r2 * 100), 2),
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "mape_percent": round(float(mape), 4) if not np.isnan(mape) else np.nan,
        "prediction_accuracy_percent": (
            round(float(prediction_accuracy), 2)
            if not np.isnan(prediction_accuracy)
            else np.nan
        ),
    }


def get_feature_names(pipeline: Pipeline, original_cols: list[str]) -> list[str]:
    """
    Gets feature names after preprocessing.
    """
    preprocessor = pipeline.named_steps["preprocess"]
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return original_cols


def save_plots(
    results_df: pd.DataFrame,
    best_model: Pipeline,
    feature_cols: list[str],
    out_dir: Path,
) -> None:
    """
    Saves model comparison and feature importance plots.
    """
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=results_df,
        x="model",
        y="test_prediction_accuracy_percent",
        hue="model",
        palette="viridis",
        legend=False,
    )
    plt.axhline(90, color="red", linestyle="--", linewidth=1, label="90% target")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Test prediction accuracy = 100 - MAPE")
    plt.xlabel("Model")
    plt.title("HCW Model Accuracy Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "model_accuracy_comparison.png", dpi=200)
    plt.close()

    final_model = best_model.named_steps["model"]

    if hasattr(final_model, "feature_importances_"):
        feature_names = get_feature_names(best_model, feature_cols)
        importances = pd.DataFrame(
            {
                "feature": feature_names[: len(final_model.feature_importances_)],
                "importance": final_model.feature_importances_,
            }
        )
        importances = importances.sort_values("importance", ascending=False).head(20)
        importances.to_csv(out_dir / "top_20_feature_importance.csv", index=False)

        plt.figure(figsize=(10, 7))
        sns.barplot(
            data=importances,
            y="feature",
            x="importance",
            hue="feature",
            palette="mako",
            legend=False,
        )
        plt.title("Top 20 Important Design and Influent Features")
        plt.xlabel("Feature importance")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(out_dir / "top_20_feature_importance.png", dpi=200)
        plt.close()


def train_and_compare(args: argparse.Namespace) -> None:
    """
    Main training workflow.
    """
    out_dir = Path(args.out)
    model_dir = out_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data, args.sheet)
    print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Target column: {args.target}")

    (
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        feature_cols,
        dropped_cols,
    ) = split_data(df, args.target)

    print(f"Training rows: {len(X_train)}")
    print(f"Validation rows: {len(X_valid)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Features used: {len(feature_cols)}")
    print(f"Columns removed to prevent leakage: {len(dropped_cols)}")

    preprocessor = make_preprocessor(X_train)
    models = build_models(preprocessor)

    results = []
    trained_models: dict[str, Pipeline] = {}

    for model_name, pipeline in models.items():
        print(f"\nTraining model: {model_name}")
        start_time = time.perf_counter()

        pipeline.fit(X_train, y_train)

        train_seconds = round(time.perf_counter() - start_time, 3)
        valid_pred = pipeline.predict(X_valid)
        test_pred = pipeline.predict(X_test)

        valid_metrics = regression_metrics(y_valid, valid_pred)
        test_metrics = regression_metrics(y_test, test_pred)

        row = {
            "model": model_name,
            "train_seconds": train_seconds,
        }
        row.update({f"valid_{key}": value for key, value in valid_metrics.items()})
        row.update({f"test_{key}": value for key, value in test_metrics.items()})

        results.append(row)
        trained_models[model_name] = pipeline

        print(
            f"{model_name} completed in {train_seconds}s | "
            f"test R2={test_metrics['r2']} | "
            f"test prediction accuracy={test_metrics['prediction_accuracy_percent']}%"
        )

    results_df = pd.DataFrame(results).sort_values("test_r2", ascending=False)
    best_name = str(results_df.iloc[0]["model"])
    best_model = trained_models[best_name]

    best_test_predictions = best_model.predict(X_test)
    predictions_df = pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "predicted": np.round(best_test_predictions, 3),
            "absolute_error": np.round(np.abs(y_test.to_numpy() - best_test_predictions), 3),
        }
    )

    results_df.to_csv(out_dir / "model_comparison_results.csv", index=False)
    predictions_df.to_csv(out_dir / "best_model_test_predictions.csv", index=False)

    joblib.dump(best_model, model_dir / "best_hcw_model.joblib")
    joblib.dump(
        {
            "target": args.target,
            "best_model": best_name,
            "feature_cols": feature_cols,
            "dropped_cols": dropped_cols,
        },
        model_dir / "metadata.joblib",
    )

    save_plots(results_df, best_model, feature_cols, out_dir)

    summary = {
        "dataset_rows": int(df.shape[0]),
        "dataset_columns": int(df.shape[1]),
        "target": args.target,
        "best_model": best_name,
        "best_test_r2": float(results_df.iloc[0]["test_r2"]),
        "best_test_r2_percent": float(results_df.iloc[0]["test_r2_percent"]),
        "best_test_prediction_accuracy_percent": float(
            results_df.iloc[0]["test_prediction_accuracy_percent"]
        ),
        "best_test_mae": float(results_df.iloc[0]["test_mae"]),
        "best_test_rmse": float(results_df.iloc[0]["test_rmse"]),
        "features_used_count": len(feature_cols),
        "columns_removed_to_prevent_leakage": dropped_cols,
        "important_note": (
            "This is a regression task. Report R2, MAE, RMSE, MAPE, and "
            "prediction accuracy = 100 - MAPE. Do not include effluent or "
            "removal columns as input features just to force high R2."
        ),
    }

    with open(out_dir / "project_run_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\n================ FINAL MODEL COMPARISON ================")
    print(results_df.to_string(index=False))
    print("========================================================")
    print(f"\nBest model: {best_name}")
    print(f"Best test R2 percentage: {summary['best_test_r2_percent']}%")
    print(
        "Best test prediction accuracy percentage: "
        f"{summary['best_test_prediction_accuracy_percent']}%"
    )
    print(f"Best model saved at: {model_dir / 'best_hcw_model.joblib'}")
    print(f"Results saved inside: {out_dir}")

    if summary["best_test_prediction_accuracy_percent"] < 90:
        print(
            "\nNote: The best no-leakage model is below 90% prediction accuracy. "
            "For your report, present R2, MAE, RMSE, MAPE, and feature importance honestly."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare 4 ML models for HCW treatment efficiency."
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATASET,
        help=(
            "Dataset path. Leave empty to auto-detect "
            "pune_hybrid_constructed_wetland_ml_dataset_*.xlsx in this folder."
        ),
    )
    parser.add_argument(
        "--sheet",
        default=DEFAULT_SHEET,
        help="Excel sheet name. Leave empty to use the first sheet.",
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help="Target column to predict.",
    )
    parser.add_argument(
        "--out",
        default="results",
        help="Folder for results, plots, predictions, and saved model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_and_compare(args)


if __name__ == "__main__":
    main()
