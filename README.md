# 🌿 Hybrid Constructed Wetland Efficiency Prediction using Machine Learning

An AI-powered Machine Learning system developed to predict the treatment efficiency of Hybrid Constructed Wetlands (HCWs) for multi-source wastewater treatment. The project compares multiple machine learning models, selects the best-performing model, and provides real-time predictions with engineering recommendations through an interactive Streamlit dashboard.

---

## 📌 Project Overview

Hybrid Constructed Wetlands are widely used for sustainable wastewater treatment. This project applies Machine Learning techniques to predict the overall treatment efficiency based on influent water quality, hydraulic parameters, and wetland design characteristics.

The application helps researchers and engineers evaluate wetland performance and supports data-driven decision-making for wastewater treatment systems.

---

## 🚀 Features

- Automatic dataset loading
- Data preprocessing and feature engineering
- Multiple Machine Learning model training
- Automatic best model selection
- Treatment efficiency prediction
- Interactive Streamlit Dashboard
- Feature Importance Analysis
- Model Performance Comparison
- Engineering Recommendations
- Prediction Confidence Analysis
- Interactive Visualizations

---

## 🤖 Machine Learning Models

- HistGradientBoosting Regressor
- Extra Trees Regressor
- Random Forest Regressor
- Fast SVM Kernel Ridge

The system automatically selects the best-performing model based on evaluation metrics.

---

## 📊 Input Parameters

- Influent BOD (mg/L)
- Influent COD (mg/L)
- Influent TSS (mg/L)
- Flow Rate (m³/day)
- Hydraulic Retention Time (days)
- Bed Area (m²)
- Bed Depth (m)
- Water Temperature (°C)
- Influent pH
- Plant Density (plants/m²)

---

## 📈 Output

The system predicts:

- Overall Treatment Efficiency (%)
- System Classification
- Operational Suitability Index
- Feature Importance
- Engineering Recommendations
- Prediction Confidence

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Matplotlib
- Joblib
- OpenPyXL

---

## 📂 Project Structure

```
Hybrid-Constructed-Wetland-Efficiency-Prediction-ML/
│
├── app.py
├── train_hcw_models.py
├── predict_hcw_efficiency.py
├── requirements.txt
├── README.md
│
├── results/
│   ├── models/
│   ├── graphs/
│   └── predictions/
│
├── dataset/
│   └── pune_hybrid_constructed_wetland_ml_dataset.xlsx
│
└── outputs/
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Hybrid-Constructed-Wetland-Efficiency-Prediction-ML.git
```

Go to the project folder:

```bash
cd Hybrid-Constructed-Wetland-Efficiency-Prediction-ML
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## 📊 Model Evaluation

The project evaluates model performance using:

- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- Prediction Accuracy

The best-performing model is automatically selected for prediction.

---

## 💡 Applications

- Wastewater Treatment Plants
- Environmental Engineering
- Smart Water Management
- Sustainable Infrastructure
- Research and Academic Projects

---

## 📷 Dashboard

The Streamlit dashboard provides:

- Research Dashboard
- Manual Prediction
- Model Comparison
- Feature Importance
- Interactive Charts
- Engineering Decision Support

---

## 📄 License

This project is developed for educational, research, and portfolio purposes only.

---
