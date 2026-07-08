import joblib
import pandas as pd
import json
from pathlib import Path

# ==================================================
# LOAD TRAINED MODEL
# ==================================================

MODEL_PATH = "results/models/best_hcw_model.joblib"
META_PATH = "results/models/metadata.joblib"
SUMMARY_PATH = "results/project_run_summary.json"

print("\nLoading trained model...")

model = joblib.load(MODEL_PATH)
metadata = joblib.load(META_PATH)

# Load summary information
if Path(SUMMARY_PATH).exists():
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        summary = json.load(f)
else:
    summary = {}

best_model = metadata.get("best_model", "Unknown")
target = metadata.get("target", "Unknown")
feature_cols = metadata.get("feature_cols", [])

model_accuracy = summary.get(
    "best_test_prediction_accuracy_percent",
    "Not Available"
)

r2_score = summary.get(
    "best_test_r2_percent",
    "Not Available"
)

# ==================================================
# USER INPUTS
# ==================================================

print("\n===================================================")
print("HYBRID CONSTRUCTED WETLAND EFFICIENCY PREDICTION")
print("===================================================")

print("\nEnter Important Parameters:\n")

user_data = {}

user_data["influent_BOD_mg_L"] = [
    float(input("Influent BOD (mg/L): "))
]

user_data["influent_COD_mg_L"] = [
    float(input("Influent COD (mg/L): "))
]

user_data["influent_TSS_mg_L"] = [
    float(input("Influent TSS (mg/L): "))
]

user_data["flow_m3_day"] = [
    float(input("Flow Rate (m3/day): "))
]

user_data["hydraulic_retention_time_day"] = [
    float(input("Hydraulic Retention Time (days): "))
]

user_data["bed_area_m2"] = [
    float(input("Bed Area (m²): "))
]

user_data["bed_depth_m"] = [
    float(input("Bed Depth (m): "))
]

user_data["water_temperature_C"] = [
    float(input("Water Temperature (°C): "))
]

user_data["influent_pH"] = [
    float(input("Influent pH: "))
]

user_data["plant_density_plants_m2"] = [
    float(input("Plant Density (plants/m²): "))
]

# ==================================================
# FILL REMAINING FEATURES
# ==================================================

for col in feature_cols:
    if col not in user_data:
        user_data[col] = [None]

# Create dataframe
X_new = pd.DataFrame(user_data)

# Ensure same column order as training
X_new = X_new.reindex(columns=feature_cols)

# ==================================================
# PREDICTION
# ==================================================

prediction = model.predict(X_new)[0]

# ==================================================
# SAVE RESULT
# ==================================================

result_df = pd.DataFrame({
    "Model_Used": [best_model],
    "Model_Accuracy_%": [model_accuracy],
    "R2_Score_%": [r2_score],
    "Predicted_Efficiency_%": [round(prediction, 2)]
})

output_file = "results/new_predictions.csv"

if Path(output_file).exists():

    old_df = pd.read_csv(output_file)

    result_df = pd.concat(
        [old_df, result_df],
        ignore_index=True
    )

result_df.to_csv(output_file, index=False)

# ==================================================
# DECISION SUPPORT REPORT
# ==================================================

# System Classification
if prediction >= 85:
    classification = "Excellent Performance"
elif prediction >= 70:
    classification = "Acceptable Performance"
else:
    classification = "Performance Improvement Required"

# Operational Suitability Index (0 to 1)
osi = round(prediction / 100, 2)

# Parameter Assessment
assessments = []

if user_data["bed_area_m2"][0] > 1000:
    assessments.append("✓ Large Bed Area")
else:
    assessments.append("⚠ Limited Bed Area")

if user_data["hydraulic_retention_time_day"][0] >= 10:
    assessments.append("✓ Adequate HRT")
else:
    assessments.append("⚠ Low Hydraulic Retention Time")

if user_data["influent_pH"][0] < 6:
    assessments.append("⚠ Acidic Influent pH")
elif user_data["influent_pH"][0] > 8.5:
    assessments.append("⚠ Alkaline Influent pH")
else:
    assessments.append("✓ Optimal Influent pH")

if user_data["bed_depth_m"][0] > 2:
    assessments.append("⚠ Excessive Bed Depth")
else:
    assessments.append("✓ Practical Bed Depth")

# Engineering Recommendations
recommendations = []

if user_data["influent_pH"][0] < 6 or user_data["influent_pH"][0] > 8.5:
    recommendations.append(
        "Maintain pH between 6.5 and 8.0 for optimum biological treatment."
    )

if user_data["plant_density_plants_m2"][0] < 10:
    recommendations.append(
        "Increase vegetation density to enhance pollutant removal."
    )

if user_data["bed_depth_m"][0] > 2:
    recommendations.append(
        "Reduce bed depth to a practical design range (0.3–1.2 m)."
    )

if user_data["hydraulic_retention_time_day"][0] < 5:
    recommendations.append(
        "Increase hydraulic retention time for better treatment efficiency."
    )

recommendations.append(
    "Continue monitoring organic loading and hydraulic performance."
)

# Confidence Level
confidence = "High"

if (
    user_data["influent_pH"][0] < 5
    or user_data["influent_pH"][0] > 9
    or user_data["bed_depth_m"][0] > 5
):
    confidence = "Medium"

if (
    user_data["bed_depth_m"][0] > 20
    or user_data["influent_pH"][0] < 4
):
    confidence = "Low"

# ==================================================
# PRINT REPORT
# ==================================================

print("\n=========================================================")
print("HYBRID CONSTRUCTED WETLAND DECISION SUPPORT REPORT")
print("=========================================================\n")

print("Model Used:")
print(best_model)

print("\nModel Performance:")
print(f"Prediction Accuracy : {model_accuracy}%")
print(f"R² Score            : {r2_score}%")

print("\nPredicted Overall Treatment Efficiency:")
print(f"{prediction:.2f} %")

print("\nSystem Classification:")
print(classification)

print("\nOperational Suitability Index:")
print(osi)

print("\nKey Design Drivers:")
print("1. Hydraulic Retention Time")
print("2. Influent BOD")
print("3. Influent COD")
print("4. Bed Area")
print("5. Water Temperature")

print("\nParameter Assessment:")
for item in assessments:
    print(item)

print("\nEngineering Recommendations:")
for rec in recommendations:
    print(f"• {rec}")

print("\nPrediction Confidence:")
print(confidence)

print("\nPrediction saved to:")
print(output_file)

print("\n=========================================================")