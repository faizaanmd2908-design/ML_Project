"""Train the Customer Credit Scoring Elastic Net model.

Run:
    python train.py
"""
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "credit_risk_dataset.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

NUM = [
    "person_age", "person_income", "person_emp_length", "loan_amnt",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
]
CAT = [
    "person_home_ownership", "loan_intent", "loan_grade",
    "cb_person_default_on_file",
]

RANDOM_STATE = 42
C = 1.0
L1_RATIO = 0.5


def load_data():
    raw = pd.read_csv(DATA_PATH)
    original_rows = len(raw)
    df = raw.drop_duplicates().copy()

    # Same domain rules used by the application.
    df = df[df["person_age"].between(18, 100)].copy()
    df.loc[df["person_emp_length"] > 60, "person_emp_length"] = np.nan

    # Derived feature: always recompute instead of trusting a stored value.
    df["loan_percent_income"] = (
        df["loan_amnt"] / df["person_income"].replace(0, np.nan)
    ).round(4)

    return df, original_rows


def make_preprocessor():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    categorical = OneHotEncoder(handle_unknown="ignore")

    return ColumnTransformer([
        ("num", numeric, NUM),
        ("cat", categorical, CAT),
    ])


def make_model():
    return Pipeline([
        ("prep", make_preprocessor()),
        ("clf", LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=C,
            l1_ratio=L1_RATIO,
            max_iter=3000,
            random_state=RANDOM_STATE,
        )),
    ])


def metrics_for(probabilities, labels, threshold):
    predictions = (probabilities >= threshold).astype(int)
    cm = confusion_matrix(labels, predictions)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities),
        "confusion": cm.tolist(),
    }


def main():
    df, original_rows = load_data()
    X = df[NUM + CAT]
    y = df["loan_status"].astype(int)

    # 60% model fitting, 20% threshold selection, 20% final held-out test.
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_fit, X_threshold, y_fit, y_threshold = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp,
        random_state=RANDOM_STATE
    )

    threshold_model = make_model().fit(X_fit, y_fit)
    threshold_prob = threshold_model.predict_proba(X_threshold)[:, 1]

    thresholds = np.linspace(0.10, 0.80, 71)
    f1_values = [
        f1_score(y_threshold, threshold_prob >= t, zero_division=0)
        for t in thresholds
    ]
    threshold = float(thresholds[int(np.argmax(f1_values))])

    # Refit the final model on all training data (80% of the dataset).
    final_model = make_model().fit(X_temp, y_temp)
    test_prob = final_model.predict_proba(X_test)[:, 1]
    test_metrics = metrics_for(test_prob, y_test, threshold)

    fpr, tpr, _ = roc_curve(y_test, test_prob)
    indices = np.linspace(0, len(fpr) - 1, min(100, len(fpr))).astype(int)

    prep = final_model.named_steps["prep"]
    clf = final_model.named_steps["clf"]
    names = prep.get_feature_names_out()
    coefficients = clf.coef_[0]

    coefficient_rows = sorted(
        (
            {"feature": n.split("__", 1)[-1], "coef": float(c)}
            for n, c in zip(names, coefficients)
        ),
        key=lambda item: -abs(item["coef"]),
    )

    transformed_train = prep.transform(X_temp)
    if hasattr(transformed_train, "toarray"):
        transformed_train = transformed_train.toarray()

    metrics = {
        "model": "Elastic Net Regularized Logistic Regression",
        "hyperparameters": {"C": C, "l1_ratio": L1_RATIO},
        "threshold": threshold,
        "threshold_selection": "Maximum F1 on a separate 20% training validation split",
        "test": test_metrics,
        "roc": {
            "fpr": fpr[indices].tolist(),
            "tpr": tpr[indices].tolist(),
        },
        "coefficients": coefficient_rows,
        "zeroed": int(np.sum(coefficients == 0)),
        "n_features": int(len(coefficients)),
        "n_train": int(len(X_temp)),
        "n_test": int(len(X_test)),
        "default_rate": float(y.mean()),
        "dataset_rows_after_cleaning": int(len(df)),
        "duplicates_removed": int(original_rows - len(df)),
    }

    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    joblib.dump(
        {
            "model": final_model,
            "threshold": threshold,
            "feature_means": transformed_train.mean(axis=0),
        },
        MODEL_DIR / "model.joblib",
    )

    print("\n=== Credit Risk Model Training Complete ===")
    print(f"Elastic Net: C={C}, l1_ratio={L1_RATIO}")
    print(f"Threshold: {threshold:.2f}")
    print("\nHeld-out test set:")
    for name in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"{name:>10}: {test_metrics[name]:.4f}")
    print("Confusion matrix:")
    print(np.array(test_metrics["confusion"]))
    print(f"\nPreprocessed features: {len(coefficients)}")
    print(f"L1-zero coefficients: {metrics['zeroed']}")
    print(f"Saved: {MODEL_DIR / 'model.joblib'}")


if __name__ == "__main__":
    main()
