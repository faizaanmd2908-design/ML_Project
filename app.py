import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

app = Flask(__name__)

ARTIFACT = joblib.load(MODEL_DIR / "model.joblib")
MODEL = ARTIFACT["model"]
THRESHOLD = float(ARTIFACT["threshold"])
FEATURE_MEANS = ARTIFACT["feature_means"]
with open(MODEL_DIR / "metrics.json", encoding="utf-8") as file:
    METRICS = json.load(file)

NUM = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]

CATS = {
    "person_home_ownership": ["RENT", "MORTGAGE", "OWN", "OTHER"],
    "loan_intent": [
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "PERSONAL",
        "DEBTCONSOLIDATION",
        "HOMEIMPROVEMENT",
    ],
    "loan_grade": list("ABCDEFG"),
    "cb_person_default_on_file": ["N", "Y"],
}

LIMITS = {
    "person_age": (18, 100),
    "person_income": (1000, 5_000_000),
    "person_emp_length": (0, 60),
    "loan_amnt": (100, 100_000),
    "cb_person_cred_hist_length": (0, 60),
    "loan_int_rate": (1, 40),
}

LABEL = {
    "person_age": "Age",
    "person_income": "Income",
    "person_emp_length": "Employment length",
    "loan_amnt": "Loan amount",
    "loan_int_rate": "Interest rate",
    "loan_percent_income": "Loan / income",
    "cb_person_cred_hist_length": "Credit history",
    "person_home_ownership": "Home ownership",
    "loan_intent": "Loan purpose",
    "loan_grade": "Loan grade",
    "cb_person_default_on_file": "Previous default",
}


def parse_application(data):
    row = {}

    for key, (low, high) in LIMITS.items():
        value = data.get(key)

        # These two fields are optional. The exact same median imputation
        # pipeline used during training handles missing values.
        if key in {"person_emp_length", "loan_int_rate"} and value in (None, ""):
            row[key] = np.nan
            continue

        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{LABEL[key]} must be a number")

        if not low <= value <= high:
            raise ValueError(f"{LABEL[key]} must be between {low:g} and {high:g}")

        row[key] = value

    for key, options in CATS.items():
        value = str(data.get(key, "")).upper()
        if value not in options:
            raise ValueError(
                f"{LABEL[key]} must be one of {', '.join(options)}"
            )
        row[key] = value

    if row["person_income"] <= 0:
        raise ValueError("Income must be greater than zero")

    # Derived feature: calculated here exactly as it is during training.
    row["loan_percent_income"] = round(
        row["loan_amnt"] / row["person_income"], 4
    )

    return row


def calculate_drivers(row):
    X = pd.DataFrame([row])
    prep = MODEL.named_steps["prep"]
    clf = MODEL.named_steps["clf"]

    transformed = prep.transform(X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    contributions = {}

    for name, value, coefficient in zip(
        prep.get_feature_names_out(),
        transformed[0] - FEATURE_MEANS,
        clf.coef_[0],
    ):
        feature = name.split("__", 1)[-1]

        # Group one-hot columns back to their original input feature.
        base_feature = next(
            (
                key
                for key in list(CATS) + NUM
                if feature == key or feature.startswith(f"{key}_")
            ),
            feature,
        )

        contributions[base_feature] = (
            contributions.get(base_feature, 0.0)
            + float(value * coefficient)
        )

    return sorted(
        (
            {
                "feature": LABEL.get(feature, feature),
                "impact": impact,
            }
            for feature, impact in contributions.items()
        ),
        key=lambda item: -abs(item["impact"]),
    )[:6]


@app.route("/")
def index():
    return render_template("index.html", cats=CATS)


@app.route("/api/metrics")
def metrics():
    return jsonify(METRICS)


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        row = parse_application(request.get_json(force=True) or {})
    except ValueError as error:
        return jsonify(error=str(error)), 400

    X = pd.DataFrame([row])
    probability = float(MODEL.predict_proba(X)[0, 1])
    high_risk = probability >= THRESHOLD

    if probability < THRESHOLD * 0.6:
        tier = "Low"
    elif probability < THRESHOLD:
        tier = "Moderate"
    elif probability < 0.6:
        tier = "High"
    else:
        tier = "Very high"

    return jsonify(
        probability=probability,
        high_risk=high_risk,
        tier=tier,
        threshold=THRESHOLD,
        used=row,
        drivers=calculate_drivers(row),
        decision=(
            "Higher predicted default risk — manual review required"
            if high_risk
            else "Lower predicted default risk — manual review recommended"
        ),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
