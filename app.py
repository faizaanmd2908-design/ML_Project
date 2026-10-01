import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request


# ---------------------------------------------------------
# APP SETUP
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates")
)


# ---------------------------------------------------------
# LOAD TRAINED MODEL
# ---------------------------------------------------------

MODEL_PATH = os.path.join(BASE_DIR, "model", "model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")

try:
    ARTIFACT = joblib.load(MODEL_PATH)

    MODEL = ARTIFACT["model"]
    THRESHOLD = float(ARTIFACT["threshold"])
    MEANS = ARTIFACT.get("means", None)

except Exception as exc:
    MODEL = None
    THRESHOLD = 0.5
    MEANS = None
    MODEL_LOAD_ERROR = str(exc)


try:
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        METRICS = json.load(f)

except Exception:
    METRICS = {}


# ---------------------------------------------------------
# FEATURES
# ---------------------------------------------------------

NUMERIC_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]

CATEGORICAL_FEATURES = {
    "person_home_ownership": [
        "RENT",
        "MORTGAGE",
        "OWN",
        "OTHER",
    ],
    "loan_intent": [
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "PERSONAL",
        "DEBTCONSOLIDATION",
        "HOMEIMPROVEMENT",
    ],
    "loan_grade": list("ABCDEFG"),
    "cb_person_default_on_file": [
        "N",
        "Y",
    ],
}


# ---------------------------------------------------------
# VALIDATION LIMITS
# ---------------------------------------------------------

LIMITS = {
    "person_age": (18, 100),
    "person_income": (1000, 5_000_000),
    "person_emp_length": (0, 60),
    "loan_amnt": (100, 100_000),
    "cb_person_cred_hist_length": (0, 60),
    "loan_int_rate": (1, 40),
}


LABELS = {
    "person_age": "Age",
    "person_income": "Annual income",
    "person_emp_length": "Employment length",
    "loan_amnt": "Loan amount",
    "loan_int_rate": "Interest rate",
    "loan_percent_income": "Loan / income ratio",
    "cb_person_cred_hist_length": "Credit history",
    "person_home_ownership": "Home ownership",
    "loan_intent": "Loan purpose",
    "loan_grade": "Loan grade",
    "cb_person_default_on_file": "Previous default",
}


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def clean_for_json(value):
    """
    Convert NumPy/Pandas values into values that are always
    safe to send as JSON.
    """

    if value is None:
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        value = float(value)

    if isinstance(value, float):
        if not np.isfinite(value):
            return None

    if pd.isna(value):
        return None

    return value


def clean_dict_for_json(data):
    """
    Recursively clean dictionary values so Flask never
    returns NaN / Infinity in JSON.
    """

    cleaned = {}

    for key, value in data.items():
        cleaned[str(key)] = clean_for_json(value)

    return cleaned


# ---------------------------------------------------------
# INPUT PARSING
# ---------------------------------------------------------

def parse_application(data):
    if not isinstance(data, dict):
        raise ValueError("Invalid request data.")

    row = {}

    # -------------------------
    # Numeric inputs
    # -------------------------

    for feature, (low, high) in LIMITS.items():

        value = data.get(feature)

        # Interest rate is optional.
        # The SAME training pipeline will impute it.
        if feature == "loan_int_rate" and value in (None, ""):
            row[feature] = np.nan
            continue

        # Employment length can also be missing.
        if feature == "person_emp_length" and value in (None, ""):
            row[feature] = np.nan
            continue

        if value in (None, ""):
            raise ValueError(
                f"{LABELS[feature]} is required."
            )

        try:
            value = float(value)

        except (TypeError, ValueError):
            raise ValueError(
                f"{LABELS[feature]} must be a number."
            )

        if not low <= value <= high:
            raise ValueError(
                f"{LABELS[feature]} must be between "
                f"{low:g} and {high:g}."
            )

        row[feature] = value

    # -------------------------
    # Categorical inputs
    # -------------------------

    for feature, options in CATEGORICAL_FEATURES.items():

        value = str(
            data.get(feature, "")
        ).strip().upper()

        if value not in options:
            raise ValueError(
                f"{LABELS[feature]} must be one of: "
                f"{', '.join(options)}."
            )

        row[feature] = value

    # -----------------------------------------------------
    # FEATURE ENGINEERING
    # -----------------------------------------------------

    income = row["person_income"]
    loan_amount = row["loan_amnt"]

    if income <= 0:
        raise ValueError(
            "Annual income must be greater than zero."
        )

    # This feature is derived automatically.
    row["loan_percent_income"] = round(
        loan_amount / income,
        4
    )

    return row


# ---------------------------------------------------------
# MODEL EXPLANATION
# ---------------------------------------------------------

def calculate_drivers(row):
    """
    Calculate approximate feature contributions for the
    linear Elastic Net model.

    The model's preprocessing is used here, so the
    explanation is based on the same transformed features
    used by the classifier.
    """

    if MODEL is None:
        return []

    try:
        if not hasattr(MODEL, "named_steps"):
            return []

        if "prep" not in MODEL.named_steps:
            return []

        if "clf" not in MODEL.named_steps:
            return []

        prep = MODEL.named_steps["prep"]
        clf = MODEL.named_steps["clf"]

        X = pd.DataFrame([row])

        transformed = prep.transform(X)

        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        transformed = np.asarray(transformed)

        coefficients = np.asarray(
            clf.coef_[0],
            dtype=float
        )

        # If means are unavailable or have the wrong size,
        # use zero-centering rather than crashing.
        if MEANS is not None:
            means = np.asarray(
                MEANS,
                dtype=float
            ).reshape(-1)

            if len(means) == transformed.shape[1]:
                centered = transformed[0] - means
            else:
                centered = transformed[0]

        else:
            centered = transformed[0]

        if len(centered) != len(coefficients):
            return []

        contributions = centered * coefficients

        feature_names = prep.get_feature_names_out()

        aggregated = {}

        for name, contribution in zip(
            feature_names,
            contributions
        ):

            # Remove transformer prefix.
            if "__" in name:
                feature = name.split(
                    "__",
                    1
                )[1]
            else:
                feature = name

            # Map transformed categorical/numeric features
            # back to their original dataset feature.
            original_feature = feature

            for candidate in (
                list(CATEGORICAL_FEATURES.keys())
                + NUMERIC_FEATURES
            ):
                if (
                    feature == candidate
                    or feature.startswith(candidate + "_")
                ):
                    original_feature = candidate
                    break

            aggregated[original_feature] = (
                aggregated.get(original_feature, 0.0)
                + float(contribution)
            )

        drivers = []

        for feature, impact in aggregated.items():

            drivers.append(
                {
                    "feature": LABELS.get(
                        feature,
                        feature
                    ),
                    "impact": float(impact),
                }
            )

        drivers.sort(
            key=lambda item: abs(item["impact"]),
            reverse=True
        )

        return drivers[:6]

    except Exception:
        # Explanation must NEVER break prediction.
        return []


# ---------------------------------------------------------
# RISK TIER
# ---------------------------------------------------------

def get_risk_tier(probability):
    if probability < THRESHOLD * 0.6:
        return "Low"

    if probability < THRESHOLD:
        return "Moderate"

    if probability < 0.6:
        return "High"

    return "Very high"


# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------

@app.route("/")
def index():
    return render_template(
        "index.html",
        cats=CATEGORICAL_FEATURES
    )


# ---------------------------------------------------------
# MODEL METRICS API
# ---------------------------------------------------------

@app.route("/api/metrics", methods=["GET"])
def metrics():

    return jsonify(METRICS)


# ---------------------------------------------------------
# PREDICTION API
# ---------------------------------------------------------

@app.route(
    "/api/predict",
    methods=["POST"]
)
def predict():

    try:

        # -------------------------------------------------
        # Check model
        # -------------------------------------------------

        if MODEL is None:
            return jsonify(
                error=(
                    "The trained model could not be loaded. "
                    f"Model error: {MODEL_LOAD_ERROR}"
                )
            ), 500

        # -------------------------------------------------
        # Read request
        # -------------------------------------------------

        data = request.get_json(
            force=True,
            silent=False
        )

        # -------------------------------------------------
        # Parse + validate
        # -------------------------------------------------

        row = parse_application(data)

        # -------------------------------------------------
        # Create DataFrame
        # -------------------------------------------------

        X = pd.DataFrame(
            [row],
            columns=NUMERIC_FEATURES
            + list(CATEGORICAL_FEATURES.keys())
        )

        # -------------------------------------------------
        # MODEL PREDICTION
        # -------------------------------------------------

        probability = float(
            MODEL.predict_proba(X)[0, 1]
        )

        # Make absolutely sure probability is valid.
        if not np.isfinite(probability):
            raise ValueError(
                "The model returned an invalid probability."
            )

        high_risk = (
            probability >= THRESHOLD
        )

        tier = get_risk_tier(
            probability
        )

        # -------------------------------------------------
        # MODEL EXPLANATION
        # -------------------------------------------------

        drivers = calculate_drivers(row)

        # -------------------------------------------------
        # DECISION-SUPPORT MESSAGE
        # -------------------------------------------------

        if high_risk:

            decision = (
                "Higher predicted default risk — "
                "manual review recommended."
            )

        else:

            decision = (
                "Lower predicted default risk — "
                "manual review recommended."
            )

        # -------------------------------------------------
        # SAFE JSON RESPONSE
        # -------------------------------------------------

        safe_row = clean_dict_for_json(row)

        safe_drivers = []

        for driver in drivers:

            safe_drivers.append(
                {
                    "feature": str(
                        driver["feature"]
                    ),
                    "impact": clean_for_json(
                        driver["impact"]
                    ),
                }
            )

        response = {
            "probability": probability,
            "high_risk": bool(high_risk),
            "tier": tier,
            "threshold": float(THRESHOLD),
            "used": safe_row,
            "drivers": safe_drivers,
            "decision": decision,
        }

        return jsonify(response), 200

    except ValueError as exc:

        return jsonify(
            error=str(exc)
        ), 400

    except Exception as exc:

        # Return the REAL error to the frontend.
        # This makes debugging much easier.
        return jsonify(
            error=(
                "Prediction failed: "
                f"{type(exc).__name__}: {exc}"
            )
        ), 500


# ---------------------------------------------------------
# GLOBAL ERROR HANDLER
# ---------------------------------------------------------

@app.errorhandler(Exception)
def handle_unexpected_error(error):

    return jsonify(
        error=(
            "Server error: "
            f"{type(error).__name__}: {error}"
        )
    ), 500


# ---------------------------------------------------------
# RUN SERVER
# ---------------------------------------------------------

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("CUSTOMER CREDIT SCORING APPLICATION")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Threshold: {THRESHOLD:.4f}")

    if MODEL is None:
        print("WARNING: Model failed to load.")
        print(MODEL_LOAD_ERROR)
    else:
        print("Model loaded successfully.")

    print("Server: http://127.0.0.1:5000")
    print("=" * 60)
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )