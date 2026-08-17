# 🍃 CO₂ Emission Predictor & R&D Recommender

An enterprise decision-support dashboard designed for automotive manufacturers to estimate vehicle CO₂ emissions during the early R&D phase and discover data-backed design alternatives to meet environmental regulations.

---

## 📌 The Problem
During the early stages of vehicle design, engineers often know core specifications (Engine Size, Vehicle Class, Transmission Type) long before physical prototypes can undergo wind-tunnel or fuel-efficiency testing. If a proposed design risks failing stringent government emission regulations, pivoting the design late in the R&D cycle costs millions of dollars.

---

## 💡 The Solution
This application solves that problem using a two-pronged machine learning approach:
1. **Predictive Modeling**: A hyperparameter-tuned **XGBoost Regression** model that predicts CO₂ emissions (Test $R^2 = 0.9509$, Test MAE: $\pm 9.10$ g/km) based purely on early-stage vehicle specifications.
2. **Recommender System**: If a proposed design's predicted emissions exceed targets, the app uses **K-Nearest Neighbors (KNN) with Cosine Similarity** to scan historical market data. It recommends similar real-world vehicles that achieved lower actual emissions, providing engineers with a blueprint for optimizing their designs.

---

## 🛠️ Tech Stack & Architecture
- **Language**: Python 3.13.2
- **Data Science & ML**: Scikit-Learn, XGBoost, Pandas, NumPy, Statsmodels
- **Interface & Viz**: Streamlit, Plotly (Gauge, Histogram, Bar)
- **Serialization**: Joblib

### Project Directory Structure
```
├── datasets/
│   ├── raw_data/
│   │   └── cars_info.csv
│   ├── cleaned_data/
│   │   └── cleaned_cars_info.csv
│   └── feature_engineered_data/
│       └── feature_engineered_cars.csv
├── notebooks/
│   ├── 01_Cleaned.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   └── 04_Model.ipynb
├── model/
│   ├── co2_production_model.pkl
│   └── co2_knn_recommender.pkl
├── pages/
│   └── 2_Recommend.py
├── app.py
├── utils.py
├── recommender.py
├── bible.md
├── requirements.txt
└── README.md
```

---

## 📊 Model Evaluation Summary
We compared 6 model classes using 5-Fold Cross-Validation and held-out test data:

| Model Class | CV R² Score | Test R² Score | Test RMSE (g/km) | Test MAE (g/km) |
|---|---|---|---|---|
| **XGBoost (Tuned - Production)** | **0.9382** | **0.9509** | **13.29** | **9.10** |
| **Random Forest (Tuned)** | 0.9343 | 0.9499 | 13.43 | 9.20 |
| **Random Forest (Untuned)** | 0.9303 | 0.9475 | 13.75 | 9.25 |
| **Decision Tree** | 0.9155 | 0.9357 | 15.21 | 9.71 |
| **Linear Regression** | 0.8404 | 0.8460 | 23.55 | 17.84 |
| **Ridge Regression** | 0.8400 | 0.8456 | 23.58 | 17.88 |
| **SGD Regression (Gradient Descent)** | 0.7776 | 0.7937 | 27.26 | 20.41 |

---

## 🚀 How to Run Locally

### 1. Set Up the Environment
Make sure you have Python 3.10+ installed. Run the following commands in your terminal (PowerShell for Windows, Bash for macOS/Linux):

**Create a virtual environment:**
```bash
python -m venv .venv
```

**Activate the virtual environment:**
* **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
* **macOS/Linux (Bash):**
  ```bash
  source .venv/bin/activate
  ```

### 2. Install Dependencies
Install all required libraries, including XGBoost, Statsmodels, and Jupyter notebook support:
```bash
pip install -r requirements.txt
```

### 3. Run the App
Launch the Streamlit web application:
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your web browser to interact with the dashboard.

---

## 📚 Study Log & Project Guide
For a deep dive into the engineering steps, data cleaning checks, outlier strategies, feature ablation details, and a mock interview preparation cheat sheet, check out **[bible.md](file:///d:/CO2_PROJECT_2.0/bible.md)**.
