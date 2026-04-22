from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from pathlib import Path

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "loan_model.pkl")

FEATURE_NAMES = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "bank_asset_value"
]

FEATURE_LABELS = {
    "cibil_score":        "CIBIL Score",
    "loan_term":          "Loan Term",
    "loan_amount":        "Loan Amount",
    "income_annum":       "Annual Income",
    "bank_asset_value":   "Bank Assets",
    "no_of_dependents":   "Dependents",
    "self_employed":      "Self Employed",
    "education":          "Education",
}

# Real importances extracted from the trained model
FEATURE_IMPORTANCES = dict(zip(FEATURE_NAMES, model.feature_importances_))


@app.route("/")
def home():
    return "ClearPath DSA Decision Engine — Running"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        dependents      = int(data.get("no_of_dependents", 0))
        education       = int(data.get("education", 0))
        self_employed   = int(data.get("self_employed", 0))
        income          = int(data.get("income_annum", 0))
        loan_amount     = int(data.get("loan_amount", 0))
        loan_term       = int(data.get("loan_term", 0))
        credit_score    = int(data.get("cibil_score", 0))
        bank_asset_value = int(data.get("bank_asset_value", 0))

        features = np.array([[
            dependents, education, self_employed, income,
            loan_amount, loan_term, credit_score, bank_asset_value
        ]])

        prediction  = model.predict(features)[0]
        proba       = model.predict_proba(features)[0]
        probability = proba.max()
        result      = "approved" if prediction == 1 else "denied"

        # --- Build feature importance payload (sorted, for Priority Queue demo) ---
        importance_payload = sorted(
            [
                {
                    "key":    k,
                    "label":  FEATURE_LABELS[k],
                    "weight": round(float(FEATURE_IMPORTANCES[k]) * 100, 2)
                }
                for k in FEATURE_NAMES
            ],
            key=lambda x: -x["weight"]
        )

        # --- DSA-structured decision trace (HashMap threshold checks) ---
        thresholds = {
            "cibil_score":      {"min": 650,  "ok": credit_score >= 650},
            "income_vs_loan":   {"ok": income >= loan_amount,
                                 "note": f"income ₹{income:,} vs loan ₹{loan_amount:,}"},
            "loan_term":        {"max": 20,   "ok": loan_term <= 20},
            "asset_coverage":   {"ok": bank_asset_value >= loan_amount * 0.3,
                                 "note": f"assets ₹{bank_asset_value:,}"},
            "dependents":       {"max": 3,    "ok": dependents < 3},
        }

        trace_steps = []
        for key, val in thresholds.items():
            trace_steps.append({
                "check": key,
                "passed": val["ok"],
                "detail": val.get("note", "")
            })

        # --- Risk flags via Min-Heap style ranking ---
        risk_items = []
        if credit_score < 500:
            risk_items.append({"factor": "Critical CIBIL score", "severity": 3})
        elif credit_score < 650:
            risk_items.append({"factor": "Low CIBIL score", "severity": 2})

        if loan_amount > income * 2:
            risk_items.append({"factor": "Loan far exceeds income", "severity": 3})
        elif loan_amount > income:
            risk_items.append({"factor": "Loan exceeds annual income", "severity": 2})

        if loan_term > 20:
            risk_items.append({"factor": "Extended loan tenure", "severity": 1})
        if self_employed:
            risk_items.append({"factor": "Variable self-employed income", "severity": 1})
        if dependents >= 4:
            risk_items.append({"factor": "High dependent burden", "severity": 2})
        elif dependents >= 3:
            risk_items.append({"factor": "Moderate dependent burden", "severity": 1})
        if bank_asset_value < loan_amount * 0.3:
            risk_items.append({"factor": "Low collateral coverage", "severity": 2})

        # Sort by severity descending (heap-ordered)
        risk_items.sort(key=lambda x: -x["severity"])

        if not risk_items:
            risk_items.append({"factor": "No significant risk factors detected", "severity": 0})

        return jsonify({
            "prediction":         result,
            "confidence":         round(probability * 100),
            "feature_importance": importance_payload,
            "trace":              trace_steps,
            "risk_items":         risk_items,
            "vote_breakdown": {
                "approved": round(float(proba[1]) * 100),
                "denied":   round(float(proba[0]) * 100)
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
