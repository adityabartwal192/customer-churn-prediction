"""
Customer Churn Prediction
=========================
End-to-end ML pipeline: EDA → Feature Engineering → Model Training → SHAP Explainability
Dataset: Synthetic telecom dataset (10,000 customers)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import joblib
import os

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_telecom_data(n=10000, seed=42):
    """Generate realistic synthetic telecom churn dataset."""
    rng = np.random.default_rng(seed)

    tenure          = rng.integers(1, 73, n)
    monthly_charges = rng.uniform(20, 120, n)
    total_charges   = monthly_charges * tenure + rng.normal(0, 50, n).clip(0)
    num_products    = rng.integers(1, 6, n)
    support_calls   = rng.integers(0, 10, n)
    contract        = rng.choice(["Month-to-month", "One year", "Two year"],
                                  n, p=[0.55, 0.25, 0.20])
    internet        = rng.choice(["DSL", "Fiber optic", "No"],
                                  n, p=[0.35, 0.45, 0.20])
    payment         = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        n, p=[0.35, 0.22, 0.22, 0.21])
    senior          = rng.choice([0, 1], n, p=[0.84, 0.16])
    partner         = rng.choice([0, 1], n, p=[0.52, 0.48])
    dependents      = rng.choice([0, 1], n, p=[0.70, 0.30])
    paperless       = rng.choice([0, 1], n, p=[0.41, 0.59])

    # Churn probability influenced by realistic drivers
    churn_prob = (
        0.05
        + 0.25 * (contract == "Month-to-month")
        + 0.15 * (internet == "Fiber optic")
        + 0.10 * (payment == "Electronic check")
        - 0.15 * (tenure > 24)
        + 0.08 * (support_calls > 5)
        - 0.05 * (num_products > 3)
        + 0.04 * senior
        + rng.normal(0, 0.05, n)
    ).clip(0.01, 0.95)
    churn = (rng.uniform(0, 1, n) < churn_prob).astype(int)

    return pd.DataFrame({
        "CustomerID":       [f"C{i:05d}" for i in range(n)],
        "Tenure":           tenure,
        "MonthlyCharges":   monthly_charges.round(2),
        "TotalCharges":     total_charges.round(2),
        "NumProducts":      num_products,
        "SupportCalls":     support_calls,
        "Contract":         contract,
        "InternetService":  internet,
        "PaymentMethod":    payment,
        "SeniorCitizen":    senior,
        "Partner":          partner,
        "Dependents":       dependents,
        "PaperlessBilling": paperless,
        "Churn":            churn
    })


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df):
    df = df.copy()
    df["AvgMonthlySpend"]  = df["TotalCharges"] / df["Tenure"].clip(lower=1)
    df["ChargePerProduct"] = df["MonthlyCharges"] / df["NumProducts"].clip(lower=1)
    df["TenureBucket"]     = pd.cut(df["Tenure"],
                                    bins=[0, 12, 24, 48, 72],
                                    labels=["<1yr", "1-2yr", "2-4yr", "4+yr"])
    df["HighSupportUser"]  = (df["SupportCalls"] >= 5).astype(int)

    # Encode categoricals
    le = LabelEncoder()
    for col in ["Contract", "InternetService", "PaymentMethod", "TenureBucket"]:
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))

    return df

FEATURE_COLS = [
    "Tenure", "MonthlyCharges", "TotalCharges", "NumProducts", "SupportCalls",
    "SeniorCitizen", "Partner", "Dependents", "PaperlessBilling",
    "AvgMonthlySpend", "ChargePerProduct", "HighSupportUser",
    "Contract_enc", "InternetService_enc", "PaymentMethod_enc", "TenureBucket_enc"
]


# ─────────────────────────────────────────────────────────────────────────────
# 3. EDA PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_eda(df, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    palette = {"Churned": "#E74C3C", "Retained": "#2E86AB"}

    df_plot = df.copy()
    df_plot["Status"] = df_plot["Churn"].map({1: "Churned", 0: "Retained"})

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Customer Churn – Exploratory Data Analysis", fontsize=16, fontweight="bold", y=0.98)

    # 1. Churn rate
    counts = df_plot["Status"].value_counts()
    axes[0,0].pie(counts, labels=counts.index, autopct="%1.1f%%",
                  colors=["#E74C3C", "#2E86AB"], startangle=90,
                  wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[0,0].set_title("Overall Churn Rate")

    # 2. Tenure distribution
    for status, grp in df_plot.groupby("Status"):
        axes[0,1].hist(grp["Tenure"], bins=30, alpha=0.65,
                       label=status, color=palette[status])
    axes[0,1].set_title("Tenure Distribution")
    axes[0,1].set_xlabel("Months")
    axes[0,1].legend()

    # 3. Monthly charges
    for status, grp in df_plot.groupby("Status"):
        axes[0,2].hist(grp["MonthlyCharges"], bins=30, alpha=0.65,
                       label=status, color=palette[status])
    axes[0,2].set_title("Monthly Charges Distribution")
    axes[0,2].set_xlabel("USD")
    axes[0,2].legend()

    # 4. Churn by contract
    churn_contract = df_plot.groupby("Contract")["Churn"].mean().sort_values(ascending=False)
    axes[1,0].bar(churn_contract.index, churn_contract.values * 100, color="#E74C3C", alpha=0.8)
    axes[1,0].set_title("Churn Rate by Contract Type")
    axes[1,0].set_ylabel("Churn Rate (%)")
    axes[1,0].tick_params(axis="x", rotation=15)

    # 5. Support calls vs churn
    churn_support = df_plot.groupby("SupportCalls")["Churn"].mean() * 100
    axes[1,1].bar(churn_support.index, churn_support.values, color="#F39C12", alpha=0.8)
    axes[1,1].set_title("Churn Rate by Support Calls")
    axes[1,1].set_xlabel("Number of Support Calls")
    axes[1,1].set_ylabel("Churn Rate (%)")

    # 6. Churn by internet service
    churn_internet = df_plot.groupby("InternetService")["Churn"].mean().sort_values(ascending=False)
    axes[1,2].bar(churn_internet.index, churn_internet.values * 100, color="#8E44AD", alpha=0.8)
    axes[1,2].set_title("Churn Rate by Internet Service")
    axes[1,2].set_ylabel("Churn Rate (%)")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/eda_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ EDA plot saved → {out_dir}/eda_overview.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(df, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)

    df_fe = engineer_features(df)
    X = df_fe[FEATURE_COLS]
    y = df_fe["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("clf",     LogisticRegression(max_iter=500, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf",     RandomForestClassifier(n_estimators=200, max_depth=8,
                                               random_state=42, n_jobs=-1))
        ]),
        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf",     GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                   learning_rate=0.05, random_state=42))
        ]),
    }

    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Model Comparison – ROC Curves", fontsize=14, fontweight="bold")
    colors = ["#2E86AB", "#E74C3C", "#27AE60"]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for (name, pipe), color, ax in zip(models.items(), colors, axes):
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        y_pred = pipe.predict(X_test)

        auc   = roc_auc_score(y_test, y_prob)
        ap    = average_precision_score(y_test, y_prob)
        cv_auc = cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc").mean()

        results[name] = {"pipeline": pipe, "auc": auc, "ap": ap, "cv_auc": cv_auc,
                         "y_prob": y_prob, "y_pred": y_pred}

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, color=color, lw=2.5, label=f"AUC = {auc:.3f}")
        ax.plot([0,1],[0,1], "k--", alpha=0.4)
        ax.fill_between(fpr, tpr, alpha=0.08, color=color)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        ax.set_xlim([0,1]); ax.set_ylim([0,1.02])

    plt.tight_layout()
    plt.savefig(f"{out_dir}/roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Print summary table
    print("\n  ┌─────────────────────────┬──────────┬──────────┬──────────────┐")
    print(  "  │ Model                   │ Test AUC │  Avg. P  │ 5-fold CV AUC│")
    print(  "  ├─────────────────────────┼──────────┼──────────┼──────────────┤")
    for name, r in results.items():
        print(f"  │ {name:<23s}  │  {r['auc']:.4f}  │  {r['ap']:.4f}  │    {r['cv_auc']:.4f}    │")
    print(  "  └─────────────────────────┴──────────┴──────────┴──────────────┘")

    return results, X_test, y_test


# ─────────────────────────────────────────────────────────────────────────────
# 5. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(results, out_dir="outputs"):
    best_name = max(results, key=lambda k: results[k]["auc"])
    best_pipe = results[best_name]["pipeline"]

    clf = best_pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    else:
        importances = np.abs(clf.coef_[0])

    feat_df = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": importances})
    feat_df = feat_df.sort_values("Importance", ascending=True).tail(12)

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(feat_df["Feature"], feat_df["Importance"],
                   color="#2E86AB", alpha=0.85)
    ax.set_title(f"Top Feature Importances — {best_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.001, bar.get_y() + bar.get_height()/2,
                f"{w:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Feature importance plot saved → {out_dir}/feature_importance.png")
    return best_name


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n═══════════════════════════════════════")
    print("  CUSTOMER CHURN PREDICTION PIPELINE  ")
    print("═══════════════════════════════════════\n")

    print("▸ Generating dataset …")
    df = generate_telecom_data(10000)
    df.to_csv("outputs/telecom_churn.csv", index=False)
    print(f"  ✓ Dataset: {len(df):,} customers, churn rate = {df['Churn'].mean():.1%}\n")

    print("▸ Running EDA …")
    plot_eda(df)

    print("\n▸ Training models …")
    results, X_test, y_test = train_and_evaluate(df)

    print("\n▸ Computing feature importances …")
    best = plot_feature_importance(results)
    print(f"  ✓ Best model: {best} (AUC = {results[best]['auc']:.4f})\n")

    print("▸ Saving best model …")
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(results[best]["pipeline"], "outputs/best_churn_model.pkl")
    print("  ✓ Model saved → outputs/best_churn_model.pkl\n")

    print("✅ Pipeline complete. Check the outputs/ folder for all artefacts.\n")
