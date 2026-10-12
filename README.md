# Customer Credit Scoring — Elastic Net Logistic Regression

A Flask web application that predicts the probability of loan default using an
Elastic Net regularized Logistic Regression model trained on a public credit-risk
dataset.

## ML pipeline

1. Load the historical credit-risk dataset.
2. Remove duplicate records.
3. Apply basic domain validation for age and employment length.
4. Recalculate `loan_percent_income` from loan amount / income.
5. Split the data into stratified training (80%) and held-out test (20%) sets.
6. Within training data, hold out 25% (i.e. 20% of the full cleaned set) for
   threshold selection; fit on the remainder.
7. Impute missing numerical values with the training-set median (with missing
   indicators) and standardize numerical features.
8. One-hot encode categorical features (`handle_unknown="ignore"`).
9. Fit Elastic Net Logistic Regression with fixed hyperparameters
   `C=1.0`, `l1_ratio=0.5` (solver `saga`).
10. Select the classification threshold by maximizing F1 on the threshold
    validation split only (no test leakage).
11. Refit on the full training portion and evaluate once on the held-out test set.
12. Save the complete preprocessing + model pipeline with joblib, together with
    the threshold and transformed feature means.
13. Serve predictions through Flask using the identical pipeline.

## Why Elastic Net?

Elastic Net combines L1 and L2 regularization.

- L1 encourages sparse coefficients and can shrink some coefficients to zero.
- L2 stabilizes coefficients, especially when predictors are correlated.
- The combination is useful for a tabular credit-risk classification problem.

## Currency / scale note

The public dataset used for training has a median annual income of approximately
55,000 and a maximum loan amount of 35,000. Those numbers are consistent with a
USD-scale consumer lending corpus. The application accepts income and loan
amounts in **the same numeric units as the training data**. Displaying a currency
symbol such as ₹ without a calibrated conversion does **not** make the model
suitable for Indian (or any other national) underwriting. Naive currency
conversion changes the relative position of an applicant in feature space and is
not validated here.

Input limits are intentionally aligned with the training support so that
standardized features remain meaningful. Extremely large values (for example a
loan of 200,000 when the training maximum is 35,000) produce out-of-distribution
standardized inputs and unreliable probabilities.

## Held-out test metrics

Reported from `model/metrics.json` (threshold 0.37 selected on validation F1):

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 86.24% |
| Precision | 69.00% |
| Recall    | 67.35% |
| F1        | 68.17% |
| ROC-AUC   | 86.95% |

## Run locally

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt
# Optional — only if you change data, preprocessing, or hyperparameters:
python train.py
python app.py
```

Open http://127.0.0.1:5000

You only need to run `train.py` again when the dataset, preprocessing, or model
configuration changes. For normal use, activate the virtual environment and run
`python app.py`.

### Tests

```bash
pytest -q
```

## API

### `GET /api/metrics`

Returns held-out test metrics, ROC curve data, confusion matrix, and model
coefficient information.

### `POST /api/predict`

Accepts applicant information as JSON and returns:

- predicted default probability
- risk tier (relative to the saved operating threshold)
- selected operating threshold
- grouped feature contributions
- neutral decision-support message

## Important ML engineering detail

The application uses the exact preprocessing pipeline saved with the trained
model. Missing interest rate and employment length values are therefore handled
by the same imputation logic during training and prediction. This avoids
training-serving skew.

The application is decision support, not an automated lending decision.
