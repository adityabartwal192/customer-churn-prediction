# 📉 Customer Churn Prediction

> End-to-end machine learning pipeline to predict telecom customer churn, identify high-risk segments, and surface the key drivers behind cancellations.

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🎯 Problem Statement

Customer acquisition costs 5–7× more than retention. Identifying customers likely to churn *before* they leave allows businesses to intervene with targeted offers, saving significant revenue. This project builds a production-ready classification pipeline on a 10,000-customer telecom dataset.

---

## 📊 Key Results

| Model | Test AUC-ROC | Avg. Precision | 5-fold CV AUC |
|---|---|---|---|
| Logistic Regression | 0.847 | 0.631 | 0.843 |
| Random Forest | 0.911 | 0.748 | 0.908 |
| **Gradient Boosting** | **0.923** | **0.771** | **0.919** |

**Best model:** Gradient Boosting with **AUC-ROC = 0.923**

---

## 🔍 Top Churn Drivers (Feature Importance)

1. **Contract type** — Month-to-month customers churn at 3× the rate of annual subscribers
2. **Tenure** — First 12 months are highest risk; churn drops sharply after 2 years
3. **Support calls** — Customers with 5+ calls in 6 months are 2× more likely to leave
4. **Monthly charges** — Higher spend correlates with higher churn in fibre plans
5. **Internet service** — Fibre optic customers churn significantly more than DSL

---

## 🗂 Project Structure

```
project1_churn_prediction/
├── churn_analysis.py        # Main pipeline (EDA → features → train → evaluate)
├── requirements.txt
├── outputs/
│   ├── telecom_churn.csv    # Generated dataset
│   ├── eda_overview.png     # EDA charts
│   ├── roc_curves.png       # ROC comparison across 3 models
│   ├── feature_importance.png
│   └── best_churn_model.pkl # Serialised best model
└── README.md
```

---

## ⚙️ Setup & Run

```bash
# Clone and install
git clone https://github.com/aaditya-bartwal/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt

# Run the full pipeline
python churn_analysis.py
```

---

## 🛠 Tech Stack

- **Python 3.9+** · **Pandas** · **NumPy**
- **Scikit-learn** — Logistic Regression, Random Forest, Gradient Boosting
- **Matplotlib** · **Seaborn** — visualisation
- **Joblib** — model serialisation

---

## 📈 Methodology

1. **Data Generation** — Synthetic 10k-row telecom dataset with realistic churn drivers
2. **EDA** — Distribution analysis, churn rates by contract/service/tenure
3. **Feature Engineering** — Tenure buckets, spend-per-product ratio, high-support flag
4. **Modelling** — 3-model comparison with 5-fold stratified cross-validation
5. **Evaluation** — AUC-ROC, Average Precision, confusion matrix, ROC curves

---

*Built as part of M.Sc. Data Science coursework — IU International University, Berlin*

