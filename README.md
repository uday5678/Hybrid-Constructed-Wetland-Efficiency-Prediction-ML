# Machine Learning Evaluation of Hybrid Constructed Wetlands

Project title: **Machine Learning Evaluation and Design Insights of Hybrid Constructed Wetlands for Multi-Source Wastewater Treatment**

This project trains and compares four fast machine-learning models to predict hybrid constructed wetland treatment performance from design, influent wastewater, hydraulic, weather, and source-mix variables.

## Dataset Check

Your dataset is suitable for this project because it contains:

- HCW design variables: wetland type, stage count, bed area, bed depth, media, vegetation, plant density
- Hydraulic/operation variables: flow, hydraulic loading rate, hydraulic retention time, pre-treatment
- Multi-source wastewater composition: domestic, industrial, agricultural, stormwater percentages
- Influent quality: pH, EC, TDS, DO, turbidity, BOD, COD, TSS, TN, NH4-N, NO3-N, TP, PO4-P, fecal coliform, oil and grease
- Effluent quality and calculated removal performance
- Outcome variables: BOD/COD/TSS/TN/TP removal, fecal coliform log removal, CPCB reuse class, overall treatment efficiency
- Reproducible split column: train, validation, test

The main target used in the code is:

```text
overall_treatment_efficiency_pct
```

## Important Note About 90% Accuracy

This is a regression problem, so the project reports:

```text
accuracy = R2 score x 100
```

A test R2 above 0.90 means the model explains more than 90% of unseen test-set variation.

Do not include `effluent_*`, `*_removal_pct`, `fecal_coliform_log_removal`, `CPCB_reuse_class`, or `maintenance_status` as input features for the main experiment. These columns reveal the answer or are closely derived from the answer, so they can create fake high accuracy.

## Models Compared

1. Histogram Gradient Boosting Regressor
2. Extra Trees Regressor
3. Random Forest Regressor
4. Fast SVM-style model using RBF Nystroem kernel approximation + Ridge regression

The fourth model is used instead of a full RBF SVR because full SVR can be slow on 18,000 rows. The approximation keeps the SVM kernel idea while running much faster.

## Folder Structure

```text
hcw_ml_project/
  train_hcw_models.py
  predict_hcw_efficiency.py
  requirements.txt
  README.md
  results/
  models/
```

## Step-by-Step Instructions

### 1. Open Command Prompt or PowerShell

Go to this folder:

```powershell
cd C:\Users\SUPRIYA\Documents\Codex\2026-05-30\is-this-dataset-which-you-gave\outputs\hcw_ml_project
```

### 2. Create a Python Environment

Use Python 3.10 or newer.

```powershell
py -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\activate
```

If `py` is not available, install Python from python.org and tick **Add Python to PATH** during installation.

### 3. Install Libraries

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Train All 4 Models

```powershell
python train_hcw_models.py
```

The script uses your dataset path by default:

```text
C:\Users\SUPRIYA\OneDrive\Desktop\UBTECH\pune_hybrid_constructed_wetland_ml_dataset_18000.xlsx
```

If the dataset is moved, pass the new path:

```powershell
python train_hcw_models.py --data "C:\path\to\dataset.xlsx"
```

### 5. Check Results

After training, open:

```text
results\model_comparison_results.csv
```

Important columns:

- `test_accuracy_percent_r2`
- `test_r2`
- `test_mae`
- `test_rmse`
- `train_seconds`

The best model is saved here:

```text
results\models\best_hcw_model.joblib
```

### 6. View Graphs

The script creates:

```text
results\model_accuracy_comparison.png
results\top_20_feature_importance.png
results\top_20_feature_importance.csv
```

Use the feature-importance plot for your design insights section.

### 7. Predict New Samples

Create a new CSV/XLSX with the same input feature columns as the dataset. Then run:

```powershell
python predict_hcw_efficiency.py --input new_samples.csv --output results\new_predictions.csv
```

## How to Write the Project Methodology

Use this structure in your report:

1. Data collection: HCW design, hydraulic, influent quality, source mix, climate/seasonal data.
2. Data cleaning: checked missing values, categorical/numerical columns, predefined train-validation-test split.
3. Leakage prevention: removed effluent and calculated removal columns from model input.
4. Feature engineering: one-hot encoding for categorical parameters and scaling for numerical parameters.
5. Model training: trained four models using the training split.
6. Model validation: selected the best model using validation R2, MAE, RMSE.
7. Final testing: reported final performance on the independent test split.
8. Design insights: interpreted feature importance to identify key HCW design and operation drivers.

## Suggested Report Objectives Mapping

- Predictive evaluation: use `overall_treatment_efficiency_pct` model.
- Pollutant-specific evaluation: rerun with targets such as `BOD_removal_pct`, `COD_removal_pct`, `TSS_removal_pct`, `TN_removal_pct`, and `TP_removal_pct`.
- Design drivers: use `top_20_feature_importance.csv`.
- Optimization suggestions: recommend favorable ranges for high-impact variables such as HRT, HLR, wetland type, media type, stage count, plant density, and source composition.

Example pollutant-specific command:

```powershell
python train_hcw_models.py --target BOD_removal_pct --out results_bod
```

## Accuracy Improvement Tips

If accuracy is below 90%:

- Keep the no-leakage experiment as the main scientific result.
- Tune only the best two models first to save time.
- Try pollutant-specific targets because some may be easier to predict than the combined efficiency score.
- Do not add effluent columns to inflate accuracy; mention this clearly in the project viva.

