# 🌲 Medical Insurance Cost Prediction Using Random Forest

<p align="center">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</p>

> A machine learning project that analyzes and predicts medical insurance expenses using **Random Forest Regression**, based on demographic and lifestyle features from the Kaggle Medical Cost Personal Dataset.


---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Model](#-model)
- [Results](#-results)
- [Key Insights](#-key-insights)
- [Conclusion](#-conclusion)

---

## 📌 Project Overview

This project applies **Random Forest Regression** to the Kaggle Medical Cost Personal Dataset to predict annual insurance expenses for individuals based on their demographic and lifestyle information.

The workflow covers:
- Data loading and cleaning
- Exploratory Data Analysis (EDA) with 6 targeted questions
- Data preprocessing and feature encoding
- Model training and evaluation
- Feature importance analysis
- Performance visualization

---

## 📊 Dataset

**Source:** [Kaggle — Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)  
**File:** `insurance.csv`

### Features

| Feature | Type | Description | Range / Values |
|---------|------|-------------|----------------|
| `age` | Numeric | Age of the primary policyholder | 18 – 64 |
| `sex` | Categorical | Biological sex | male / female |
| `bmi` | Numeric | Body Mass Index (weight/height²) | 16.0 – 53.1 |
| `children` | Numeric | Number of dependents on the policy | 0 – 5 |
| `smoker` | Categorical | Whether the individual smokes tobacco | yes / no |
| `region` | Categorical | US residential region | northeast / northwest / southeast / southwest |
| `expenses` | Numeric *(target)* | Annual medical insurance expenses (USD) | $1,122 – $63,770 |

### Dataset Statistics

```
Shape:   1338 rows × 7 columns  →  1337 rows after removing 1 duplicate
Missing: 0 values across all columns
```

| Statistic | age | bmi | children | expenses |
|-----------|-----|-----|----------|----------|
| Mean | 39.22 | 30.67 | 1.10 | $13,279 |
| Std | 14.04 | 6.10 | 1.21 | $12,110 |
| Min | 18 | 16.0 | 0 | $1,122 |
| Median | 39 | 30.4 | 1 | $9,386 |
| Max | 64 | 53.1 | 5 | $63,770 |

---

## 📁 Project Structure

```
medical-insurance-random-forest-project/
│
├── medical_insurance_cost_prediction_random_forest.ipynb   # Main notebook
├── insurance.csv                                           # Dataset
└── README.md                                               # This file
└── RF_Insurance_Final_v3.pptx                              # The presntation
```

---

## ⚙️ Installation

### Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

### Run the Notebook

```bash
# Clone the repository
git clone https://github.com/Yasir-Aladwani/medical-insurance-random-forest-project.git
cd medical-insurance-random-forest-project

# Launch Jupyter
jupyter notebook medical_insurance_cost_prediction_random_forest.ipynb
```

Or open directly in Google Colab using the badge at the top.

---

## 🔍 Exploratory Data Analysis

Six EDA questions were investigated to understand the data before modelling.

---

### EDA Q1 — What is the distribution of medical insurance expenses?

The distribution is **right-skewed (positively skewed)**. Most individuals pay relatively low insurance costs, while a smaller number of high-risk individuals incur very high charges — creating a long right tail that extends to ~$64,000.

---

### EDA Q2 — Does age affect insurance expenses?

There is a **positive relationship** between age and expenses: costs generally rise as individuals get older. Importantly, the scatter plot reveals **three distinct cost bands**, suggesting another factor (smoking status) creates parallel subgroups that each increase with age independently.

---

### EDA Q3 — Do smokers pay higher insurance expenses?

Yes — significantly so.

| Group | Average Annual Expenses |
|-------|------------------------|
| Non-Smokers | **$8,441** |
| Smokers | **$32,050** |

Smokers pay **3.8× more** on average. The correlation between `smoker` and `expenses` is **0.79** — the strongest of any feature in the dataset.

---

### EDA Q4 — Does BMI affect insurance expenses?

Higher BMI is associated with higher insurance costs, but the relationship is **not linear**. The presence of a dense high-cost cluster above BMI = 30 points to a strong interaction between obesity and smoking status — obese smokers represent the highest-cost subgroup.

---

### EDA Q5 — Are there outliers in medical insurance expenses?

**139 outliers** were identified using the IQR method. These were **retained** — they represent genuine high-cost medical cases, not data entry errors, and removing them would introduce bias into the model.

---

### EDA Q6 — Which variable is most associated with expenses?

Correlation with `expenses`:

| Feature | Correlation |
|---------|-------------|
| `smoker` | **0.787** |
| `age` | 0.298 |
| `bmi` | 0.199 |
| `children` | 0.067 |
| `sex` | 0.058 |
| `region` | -0.007 |

`smoker` is by far the strongest predictor. `age` and `bmi` follow as meaningful secondary predictors.

---

## 🤖 Model

### Why Random Forest?

Random Forest is an ensemble learning algorithm that combines many decision trees to improve accuracy and reduce overfitting. It is well-suited to this dataset because:

- It captures **non-linear relationships** between features and the target
- It handles **mixed data types** (numeric + categorical) without scaling
- It provides **built-in feature importance** scores
- It is **resistant to overfitting** due to the averaging effect

### Preprocessing

| Step | Action |
|------|--------|
| Duplicate removal | 1 row removed → 1,337 rows |
| Missing values | None — no imputation required |
| Encoding | `sex`, `smoker`, `region` label-encoded via `LabelEncoder` |
| Feature scaling | Not applied (tree-based model, not distance-based) |
| Train/test split | 80% train / 20% test, `random_state=42` |

### Model Configuration

```python
rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
```

| Hyperparameter | Value | Notes |
|---------------|-------|-------|
| `n_estimators` | `200` | Number of trees in the forest |
| `max_depth` | `None` | Trees grow fully — no pruning |
| `max_features` | `'sqrt'` | sklearn default for regression |
| `random_state` | `42` | Reproducibility |

---

## 📈 Results

### Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **MAE** | **$2,545** | Average prediction error — predictions are off by ~$2.5k on a target ranging $1k–$64k |
| **RMSE** | **$4,618** | Penalises large errors more; reflects a small number of high-cost outliers |
| **R²** | **0.884** | The model explains **88.4%** of the variance in insurance expenses |

### Feature Importance

| Feature | Importance Score |
|---------|-----------------|
| `smoker` | **0.599 (59.9%)** |
| `bmi` | 0.215 (21.5%) |
| `age` | 0.138 (13.8%) |
| `children` | 0.024 (2.4%) |
| `region` | 0.017 (1.7%) |
| `sex` | 0.008 (0.8%) |

Smoking status alone accounts for nearly **60%** of the model's predictive power.

---

## 💡 Key Insights

1. **Lifestyle factors dominate cost** — Smoking and BMI together drive over 80% of the model's predictions. Insurance pricing is primarily shaped by modifiable lifestyle choices, not fixed demographics.

2. **Non-linear relationships matter** — The smoker × BMI interaction is fundamentally non-linear. Linear regression achieves only R² ≈ 0.75 on this dataset, versus Random Forest's 0.884 — an 18-point gap explained entirely by the model's ability to capture these interactions.

3. **Outliers are real, not noise** — The 139 high-cost outliers are genuine high-risk individuals. Retaining them allowed the model to learn the true distribution of insurance costs.

4. **Age creates distinct risk bands** — The three horizontal clusters visible in the age scatter plot represent non-smokers (low cost), light risk (mid cost), and smokers (high cost) — each rising independently with age.

5. **Gender and region are negligible** — `sex` (0.8%) and `region` (1.7%) have minimal impact, confirming that geographic and demographic factors are largely irrelevant once lifestyle variables are accounted for.

---

## ✅ Conclusion

A Random Forest Regression model was successfully trained to predict medical insurance expenses with an **R² of 0.884** and an average prediction error (MAE) of **$2,545**. The results confirm that **smoking status, BMI, and age** are the primary determinants of insurance cost, while demographic variables such as sex and region contribute negligibly.

The model's performance demonstrates the power of ensemble methods for capturing complex, non-linear interactions in real-world healthcare cost data — and highlights actionable public health implications: wellness programs targeting smoking cessation and weight management would have the largest measurable impact on reducing insurance expenses.

---

## 📄 License

This project is for educational purposes as part of the Tuwaiq Academy Machine Learning programme.
