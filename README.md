# Customer Credit Scoring — Elastic Net Logistic Regression

A Flask web application that predicts the probability of loan default using an
Elastic Net regularized Logistic Regression model trained on the Credit Risk
Dataset.

## ML pipeline

1. Load the historical credit-risk dataset.
2. Remove duplicate records.
3. Apply basic domain validation for age and employment length.
4. Recalculate `loan_percent_income` from loan amount / income.
5. Split the data into stratified training and held-out test sets.
6. Impute missing numerical values with the training-set median.
7. Add missing-value indicators and standardize numerical features.
8. One-hot encode categorical features.
9. Tune Elastic Net Logistic Regression with 5-fold stratified cross-validation.
10. Select the classification threshold using only out-of-fold training predictions.
11. Evaluate once on the held-out test set.
12. Save the complete preprocessing + model pipeline with joblib.
13. Serve predictions through Flask.

## Why Elastic Net?

Elastic Net combines L1 and L2 regularization.

- L1 encourages sparse coefficients and can shrink some coefficients to zero.
- L2 stabilizes coefficients, especially when predictors are correlated.
- The combination is useful for a tabular credit-risk classification problem.

## Run locally

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

You only need to run `train.py` again when the dataset, preprocessing, or model
configuration changes. For normal use, activate the virtual environment and run
`python app.py`.

## API

### `GET /api/metrics`

Returns held-out test metrics, ROC curve data, confusion matrix, and model
coefficient information.

### `POST /api/predict`

Accepts applicant information as JSON and returns:

- predicted default probability
- risk tier
- selected operating threshold
- grouped feature contributions
- neutral decision-support message

## Important ML engineering detail

The application uses the exact preprocessing pipeline saved with the trained
model. Missing interest rate and employment length values are therefore handled
by the same imputation logic during training and prediction. This avoids
training-serving skew.

The application is decision support, not an automated lending decision.
