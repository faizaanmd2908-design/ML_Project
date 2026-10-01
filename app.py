import json, joblib, numpy as np, pandas as pd
from flask import Flask, jsonify, render_template, requestf

app = Flask(__name__)
art = joblib.load("model/model.joblib"); MODEL, THR, GRATE, MEANS = art["model"], art["threshold"], art["grade_rate"], art["means"]
METRICS = json.load(open("model/metrics.json"))
NUM = ["person_age","person_income","person_emp_length","loan_amnt","loan_int_rate",
       "loan_percent_income","cb_person_cred_hist_length"]
CATS = {"person_home_ownership": ["RENT","MORTGAGE","OWN","OTHER"],
        "loan_intent": ["EDUCATION","MEDICAL","VENTURE","PERSONAL","DEBTCONSOLIDATION","HOMEIMPROVEMENT"],
        "loan_grade": list("ABCDEFG"), "cb_person_default_on_file": ["N","Y"]}
LIM = {"person_age": (18,100), "person_income": (1000,5e6), "person_emp_length": (0,60),
       "loan_amnt": (100,100000), "cb_person_cred_hist_length": (0,60), "loan_int_rate": (1,40)}
LABEL = {"person_age":"Age","person_income":"Income","person_emp_length":"Employment length",
  "loan_amnt":"Loan amount","loan_int_rate":"Interest rate","loan_percent_income":"Loan / income",
  "cb_person_cred_hist_length":"Credit history","person_home_ownership":"Home ownership",
  "loan_intent":"Loan purpose","loan_grade":"Loan grade","cb_person_default_on_file":"Prior default"}

def parse(d):
    row = {}
    for k,(lo,hi) in LIM.items():
        v = d.get(k)
        if k in ("person_emp_length","loan_int_rate") and v in (None, ""): row[k] = np.nan; continue
        try: v = float(v)
        except (TypeError, ValueError): raise ValueError(f"{LABEL[k]} must be a number")
        if not lo <= v <= hi: raise ValueError(f"{LABEL[k]} must be between {lo:g} and {hi:g}")
        row[k] = v
    for k, opts in CATS.items():
        v = str(d.get(k, "")).upper()
        if v not in opts: raise ValueError(f"{LABEL[k]} must be one of {', '.join(opts)}")
        row[k] = v
    
    row["loan_percent_income"] = round(row["loan_amnt"] / row["person_income"], 4)
    return row

@app.route("/")
def index(): return render_template("index.html", cats=CATS)

@app.route("/api/metrics")
def metrics(): return jsonify(METRICS)

@app.route("/api/predict", methods=["POST"])
def predict():
    try: row = parse(request.get_json(force=True) or {})
    except ValueError as e: return jsonify(error=str(e)), 400
    X = pd.DataFrame([row]); p = float(MODEL.predict_proba(X)[0, 1])
    prep, clf = MODEL.named_steps["prep"], MODEL.named_steps["clf"]
    x = prep.transform(X); x = x.toarray() if hasattr(x, "toarray") else x
    contrib = {}
    for n, v, c in zip(prep.get_feature_names_out(), x[0]-MEANS, clf.coef_[0]):
        f = n.split("__",1)[1]
        f = next((k for k in list(CATS)+NUM if f.startswith(k) or f.endswith(k)), f)
        contrib[f] = contrib.get(f, 0) + float(v*c)
    drivers = sorted(({"feature": LABEL.get(k,k), "impact": v} for k,v in contrib.items()),
                     key=lambda d: -abs(d["impact"]))[:6]
    tier = "Low" if p < THR*0.6 else "Moderate" if p < THR else "High" if p < 0.6 else "Very high"
    return jsonify(probability=p, high_risk=p >= THR, tier=tier, threshold=THR,
                   used=row, drivers=drivers,
                   decision="Recommend review / decline" if p >= THR else "Likely to repay - approve")

if __name__ == "__main__": app.run(host="0.0.0.0", port=5000, debug=False)
