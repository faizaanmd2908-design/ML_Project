# Credit Risk Scorer (Elastic Net Logistic Regression)
Windows 11 / Ubuntu / WSL, Python 3.10+.
```
pip install -r requirements.txt
python train.py      # trains, tunes, saves model/ (about 1-2 min)
python app.py        # open http://127.0.0.1:5000
```
- Cleaning: duplicates, age>100 and tenure>60 removed/blanked, income capped at 99.5th pct; median imputation + missing flags; one-hot categoricals; spline basis on numerics.
- Model: LogisticRegression(elasticnet, saga), GridSearchCV over C and l1_ratio (5-fold, ROC-AUC). Cut-off picked on out-of-fold train predictions; test set touched once.
- Site: risk %, tier, decision, per-feature drivers vs. the average applicant, ROC curve, confusion matrix, top coefficients, gradient-boosting baseline for context.
- API: POST /api/predict (JSON), GET /api/metrics. Interest rate blank = median for that grade.
