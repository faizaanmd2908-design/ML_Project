"""Train Elastic Net logistic regression for credit risk. Run: python train.py"""
import warnings; warnings.filterwarnings("ignore")
import json, joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

NUM = ["person_age","person_income","person_emp_length","loan_amnt","loan_int_rate",
       "loan_percent_income","cb_person_cred_hist_length"]
CAT = ["person_home_ownership","loan_intent","loan_grade","cb_person_default_on_file"]
df = pd.read_csv("credit_risk_dataset.csv").drop_duplicates()
df = df[df.person_age <= 100].copy()                       # impossible ages
df.loc[df.person_emp_length > 60, "person_emp_length"] = np.nan  # impossible tenure
df["person_income"] = df.person_income.clip(upper=df.person_income.quantile(.995))
X, y = df[NUM+CAT], df.loan_status
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)

def prep(spline=False):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())] + ([("sp", SplineTransformer(n_knots=4, degree=3, include_bias=False))] if spline else [])), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])

pipe = Pipeline([("prep", prep(spline=True)), ("clf", LogisticRegression(
    penalty="elasticnet", solver="saga", max_iter=5000, random_state=42))])
grid = {"clf__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10], "clf__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]}
cv = StratifiedKFold(5, shuffle=True, random_state=42)
gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1).fit(Xtr, ytr)
best = gs.best_estimator_
print("Best params:", gs.best_params_, "CV AUC: %.4f" % gs.best_score_)

# decision threshold: maximise F1 on out-of-fold train predictions (no test leakage)
oof = cross_val_predict(best, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
ths = np.linspace(.1, .8, 71); thr = float(ths[np.argmax([f1_score(ytr, oof >= t) for t in ths])])

def evaluate(p, t=.5):
    pr = (p >= t).astype(int); cm = confusion_matrix(yte, pr)
    return dict(accuracy=accuracy_score(yte, pr), precision=precision_score(yte, pr),
                recall=recall_score(yte, pr), f1=f1_score(yte, pr), roc_auc=roc_auc_score(yte, p),
                confusion=cm.tolist())
p_te = best.predict_proba(Xte)[:, 1]
main = evaluate(p_te, thr)
fpr, tpr, _ = roc_curve(yte, p_te); idx = np.linspace(0, len(fpr)-1, 80).astype(int)

hgb = Pipeline([("prep", prep()), ("clf", HistGradientBoostingClassifier(random_state=42))]).fit(Xtr, ytr)
base = evaluate(hgb.predict_proba(Xte)[:, 1], .5)
names = best.named_steps["prep"].get_feature_names_out()
coef = best.named_steps["clf"].coef_[0]
coefs = sorted(({"feature": n.split("__",1)[1], "coef": float(c)} for n, c in zip(names, coef)),
               key=lambda d: -abs(d["coef"]))
grade_rate = df.groupby("loan_grade").loan_int_rate.median().round(2).to_dict()
metrics = dict(best_params=gs.best_params_, cv_auc=gs.best_score_, threshold=thr, test=main,
    baseline_gradient_boosting=base, roc={"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist()},
    coefficients=coefs, zeroed=int((coef == 0).sum()), n_features=len(coef),
    n_train=len(Xtr), n_test=len(Xte), default_rate=float(y.mean()),
    grade_rate=grade_rate, ranges={c: [float(df[c].min()), float(df[c].max())] for c in NUM})
json.dump(metrics, open("model/metrics.json", "w"), indent=1)
_m = best.named_steps["prep"].transform(Xtr); _m = _m.toarray() if hasattr(_m, "toarray") else _m
joblib.dump({"means": _m.mean(0), "model": best, "threshold": thr, "grade_rate": grade_rate}, "model/model.joblib")
print(json.dumps({k: round(v, 4) for k, v in main.items() if k != "confusion"}, indent=1))
print("Baseline GB AUC: %.4f | zeroed coefs: %d/%d" % (base["roc_auc"], metrics["zeroed"], len(coef)))
