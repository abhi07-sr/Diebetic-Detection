🩺 Diabetes Detection & Predictive Analysis (Machine Learning)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-v2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-v1.24%2B-013243.svg)](https://numpy.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

A comprehensive End-to-End Machine Learning project aimed at the early detection and risk prediction of diabetes mellitus using clinical diagnostic measurements from the **Pima Indians Diabetes Database**.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset Description](#-dataset-description)
- [Data Preprocessing & EDA](#-data-preprocessing--eda)
- [Machine Learning Models](#-machine-learning-models)
- [Model Evaluation & Comparison](#-model-evaluation--comparison)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Model Inference (How to Use)](#-model-inference-how-to-use)
- [Key Insights & Future Work](#-key-insights--future-work)
- [Author & Acknowledgments](#-author--acknowledgments)

---

## 🔬 Project Overview

Diabetes is a chronic metabolic condition characterized by elevated blood glucose levels, leading over time to serious damage to the heart, blood vessels, eyes, kidneys, and nerves. Early and accurate detection enables proactive lifestyle interventions and targeted medical treatment.

This project implements a complete data science workflow:
1. **Exploratory Data Analysis (EDA)** to understand distributions, feature interactions, and clinical anomalies.
2. **Medical Anomaly & Missing Value Handling** (biologically impossible zero values in glucose, blood pressure, etc.).
3. **Feature Scaling & Outlier Treatment** using standard scaling and quantile bounds.
4. **Synthetic Oversampling (SMOTE)** to handle class imbalance between diabetic and non-diabetic cases.
5. **Model Benchmarking**: Comparing **Logistic Regression**, **Gaussian Naive Bayes** (both Scikit-Learn and pure mathematical from-scratch implementations), and **K-Nearest Neighbors (KNN)**.
6. **Model Persistence**: Serializing the best-performing model (`.pkl`) for production inference.

---

## 📊 Dataset Description

The dataset used is the **Pima Indians Diabetes Database** (`diabetes.csv`), sourced originally from the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK).

- **Total Instances:** 768 patient records
- **Total Attributes:** 8 clinical predictor features + 1 binary target label

### Feature Dictionary:
| Feature | Type | Description |
| :--- | :--- | :--- |
| `Pregnancies` | Numerical (Integer) | Number of times pregnant |
| `Glucose` | Numerical (Continuous) | Plasma glucose concentration at 2 hours in an oral glucose tolerance test |
| `BloodPressure` | Numerical (Continuous) | Diastolic blood pressure ($mm\ Hg$) |
| `SkinThickness` | Numerical (Continuous) | Triceps skin fold thickness ($mm$) |
| `Insulin` | Numerical (Continuous) | 2-Hour serum insulin ($\mu U/ml$) |
| `BMI` | Numerical (Continuous) | Body Mass Index ($weight\ in\ kg / (height\ in\ m)^2$) |
| `DiabetesPedigreeFunction` | Numerical (Continuous) | Genetic diabetes pedigree score representing diabetes family history |
| `Age` | Numerical (Integer) | Age in years |
| **`Outcome`** | **Binary Target** | **0 = Non-Diabetic**, **1 = Diabetic** |

---

## 🛠️ Data Preprocessing & EDA

### 1. Handling Biologically Impossible Zeroes
In the raw dataset, several physiological variables contained `0` values, which are clinically invalid (a living individual cannot have 0 blood pressure, 0 glucose, or 0 BMI):
- **Glucose**, **BloodPressure**, **BMI**: Replaced `0` with the **mean** / **median** depending on skewness.
- **Insulin**, **SkinThickness**, **Pregnancies**: Replaced `0` with the **median** due to right-skewed distributions.

### 2. Outlier Detection & Treatment
- Generated distribution plots (`sns.displot` / `kde`) and box plots (`boxplot.png`) to identify extreme deviations.
- Capped high extreme outliers (e.g., upper 95th percentile filtering on `Insulin`) to prevent gradient distortion and high variance.

### 3. Feature Standardization
- Applied `StandardScaler` from Scikit-Learn:
  $$\mu = 0, \quad \sigma = 1$$
  Ensuring distance-sensitive models (KNN, Logistic Regression) give balanced weight to all clinical features.

### 4. Class Imbalance Handling
- The raw dataset had ~65% non-diabetic (0) and ~35% diabetic (1) patients.
- Applied **SMOTE (Synthetic Minority Over-sampling Technique)** on the training set to prevent model bias towards the majority class:
  ```python
  from imblearn.over_sampling import SMOTE
  smote = SMOTE(random_state=42)
  X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
  ```

---

## 🧠 Machine Learning Models

### 1. Logistic Regression (Primary Model)
- Fits a sigmoid activation curve to predict class probabilities:
  $$P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \dots + \beta_n X_n)}}$$
- Highly interpretable, well-calibrated probabilities for clinical risk stratification.
- Exported as `logistic_regression_model.pkl` / `classification_model.pkl`.

### 2. Gaussian Naive Bayes (GNB)
- Evaluated both using `sklearn.naive_bayes.GaussianNB` and a **from-scratch mathematical implementation** utilizing the Gaussian probability density function:
  $$P(x_i | y) = \frac{1}{\sqrt{2\pi\sigma_y^2}} \exp\left(-\frac{(x_i - \mu_y)^2}{2\sigma_y^2}\right)$$
- Operates under conditional independence assumption, robust and computationally fast.

### 3. K-Nearest Neighbors (KNN)
- Non-parametric distance-based classification based on feature proximity in Euclidean space.

---

## 📈 Model Evaluation & Comparison

Models were evaluated on a dedicated hold-out test set ($33\%$ test split, $N = 238$ test samples):

| Model | Accuracy | Non-Diabetic Precision | Diabetic Precision | Diabetic Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** 🏆 | **~75%** | **0.85** | **0.60** | **0.72** | **0.73** |
| **Gaussian Naive Bayes** | ~72% | 0.82 | 0.57 | 0.66 | 0.69 |
| **K-Nearest Neighbors (KNN)** | ~68% | 0.84 | 0.52 | 0.76 | 0.67 |

> [!NOTE]
> In medical diagnostics, **Recall (Sensitivity)** for the diabetic class is paramount because a **False Negative** (failing to identify an actual diabetic patient) carries higher risk than a False Positive. **Logistic Regression** and **KNN** provided high sensitivity while Logistic Regression maintained superior overall precision and generalizability.

---

## 📁 Project Structure

```text
DIBETIES DETECTION/
│
├── README.md                                          <- Project documentation and overview
│
└── Hands On_ Predictive Analysis in Diabetes/
    ├── diabetes.csv                                   <- Clinical Pima Indians dataset
    ├── Hands On_ Predictive Analysis in Diabetes.ipynb<- Complete EDA, SMOTE, & Multi-model benchmark
    ├── Logistic REG For Diebetics Detection-1.ipynb   <- Detailed Logistic Regression & outlier pipeline
    ├── logistic_regression_model.pkl                  <- Serialized trained model ready for deployment
    ├── heatmap.png                                    <- Feature correlation heatmap
    └── boxplot.png                                    <- Outlier visualization before/after scaling
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9+
- Recommended: JupyterLab, VS Code, or Google Colab

### 1. Clone or Navigate to the Directory
```bash
cd "DIBETIES DETECTION/Hands On_ Predictive Analysis in Diabetes"
```

### 2. Set Up a Virtual Environment (Optional but Recommended)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
```

### 4. Run the Notebooks
Launch Jupyter Notebook or Jupyter Lab:
```bash
jupyter notebook
```
Open `Hands On_ Predictive Analysis in Diabetes.ipynb` or `Logistic REG For Diebetics Detection-1.ipynb` and run all cells.

---

## 💻 Model Inference (How to Use)

To load the trained model and make predictions on new patient clinical data:

```python
import pickle
import numpy as np

# 1. Load the trained model
with open("logistic_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

# 2. Sample patient data:
# [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
# Note: Input must be scaled using the same StandardScaler parameters as training!
sample_patient = np.array([[2, 130, 70, 25, 120, 28.5, 0.45, 35]])

# 3. Predict class (0 = Non-Diabetic, 1 = Diabetic)
prediction = model.predict(sample_patient)
prediction_prob = model.predict_proba(sample_patient)

status = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
print(f"Prediction: {status}")
print(f"Risk Probability: {prediction_prob[0][1]*100:.2f}%")
```

---

## 🔮 Key Insights & Future Work

### Clinical Takeaways
- **Glucose concentration** and **BMI** exhibited the highest positive correlation with diabetic status.
- **Age** and **Pregnancies** showed secondary correlation, indicating increased susceptibility with age.

### Potential Extensions
- [ ] Implement hyperparameter tuning (GridSearchCV, Optuna) on ensemble models (Random Forest, XGBoost, LightGBM).
- [ ] Build an interactive web application using **Streamlit** or a **FastAPI** REST microservice.
- [ ] Integrate SHAP (SHapley Additive exPlanations) for model explainability and clinician-facing feature importance dashboards.

---

## 👤 Author & Acknowledgments
- **Project Developer:** Abhinandan Naik
- **Dataset:** National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) / Pima Indian Community
- **Libraries Used:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn, Imbalanced-Learn
