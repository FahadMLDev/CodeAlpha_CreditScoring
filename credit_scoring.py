"""
CodeAlpha Task 1: Credit Scoring Model
Dataset: "Give Me Some Credit" (Kaggle) -> keep cs-training.csv in the same folder.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix,
                             classification_report)

# ---------- Step 1: Load data ----------
df = pd.read_csv("cs-training.csv")
df = df.drop(columns=["Unnamed: 0"], errors="ignore")
TARGET = "SeriousDlqin2yrs"          # 1 = defaulted / seriously delinquent
print("Shape:", df.shape)
print("Class balance:\n", df[TARGET].value_counts(normalize=True))

# ---------- Step 2: Clean data ----------
df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
df["NumberOfDependents"] = df["NumberOfDependents"].fillna(0)
df = df[(df["age"] > 18) & (df["age"] < 100)]
for col in ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "MonthlyIncome"]:
    df[col] = df[col].clip(upper=df[col].quantile(0.99))
for col in ["NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTimes90DaysLate",
            "NumberOfTime60-89DaysPastDueNotWorse"]:
    df[col] = df[col].clip(upper=20)   # 96/98 are dataset error codes

# ---------- Step 3: Feature engineering ----------
late30 = "NumberOfTime30-59DaysPastDueNotWorse"
late60 = "NumberOfTime60-89DaysPastDueNotWorse"
late90 = "NumberOfTimes90DaysLate"
df["TotalLatePayments"] = df[late30] + df[late60] + df[late90]
df["HasLatePayment"] = (df["TotalLatePayments"] > 0).astype(int)
df["IncomePerPerson"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)
df["MonthlyDebt"] = df["DebtRatio"] * df["MonthlyIncome"]
df["LoansPerAge"] = df["NumberOfOpenCreditLinesAndLoans"] / df["age"]

# ---------- Step 4: Split ----------
X = df.drop(columns=[TARGET])
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# ---------- Step 5: Models (class_weight handles imbalance) ----------
models = {
    "Logistic Regression": Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=50, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=20,
        class_weight="balanced", n_jobs=-1, random_state=42),
}

# ---------- Step 6: Train & evaluate ----------
results, probas = [], {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    probas[name] = proba
    results.append({
        "Model": name,
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
    })
    print(f"\n=== {name} ===")
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))

res = pd.DataFrame(results).set_index("Model").round(3)
print("\nModel comparison:\n", res)
best = res["ROC-AUC"].idxmax()
print("Best model by ROC-AUC:", best)

# ---------- Step 7: Plots ----------
plt.figure(figsize=(6, 5))
for name, p in probas.items():
    fpr, tpr, _ = roc_curve(y_test, p)
    plt.plot(fpr, tpr, label=f"{name} (AUC={res.loc[name, 'ROC-AUC']})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves"); plt.legend(); plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150)

rf = models["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
plt.figure(figsize=(7, 6)); imp.plot(kind="barh")
plt.title("Random Forest Feature Importance"); plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)