# ================================
# IMPORTS
# ================================
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import shap

app = Flask(__name__)

# ================================
# LOAD MODELS
# ================================
xgb_model = joblib.load('xgb_model.pkl')
iso_model  = joblib.load('iso_model.pkl')
X_test     = pd.read_csv('X_test.csv')

# SHAP Explainer (for XGBoost)
explainer = shap.TreeExplainer(xgb_model)

# ================================
# PREPROCESS FUNCTION
# ================================
def preprocess(step, amount, oldbalanceOrg, newbalanceOrig,
               oldbalanceDest, newbalanceDest, txn_type):

    type_CASH_IN  = 1 if txn_type == "CASH_IN"  else 0
    type_CASH_OUT = 1 if txn_type == "CASH_OUT" else 0
    type_DEBIT    = 1 if txn_type == "DEBIT"    else 0
    type_PAYMENT  = 1 if txn_type == "PAYMENT"  else 0
    type_TRANSFER = 1 if txn_type == "TRANSFER" else 0

    log_amount             = np.log1p(amount)
    orig_emptied           = 1 if (oldbalanceOrg > 0 and newbalanceOrig == 0) else 0
    dest_unchanged         = 1 if (oldbalanceDest == newbalanceDest) else 0
    amount_vs_orig_balance = amount / (oldbalanceOrg + 1)
    balance_diff_orig      = oldbalanceOrg - newbalanceOrig
    balance_diff_dest      = newbalanceDest - oldbalanceDest
    amount_mismatch        = abs(balance_diff_orig - amount)

    return pd.DataFrame([{
        'step'                  : step,
        'amount'                : amount,
        'log_amount'            : log_amount,
        'oldbalanceOrg'         : oldbalanceOrg,
        'newbalanceOrig'        : newbalanceOrig,
        'oldbalanceDest'        : oldbalanceDest,
        'newbalanceDest'        : newbalanceDest,
        'type_CASH_IN'          : type_CASH_IN,
        'type_CASH_OUT'         : type_CASH_OUT,
        'type_DEBIT'            : type_DEBIT,
        'type_PAYMENT'          : type_PAYMENT,
        'type_TRANSFER'         : type_TRANSFER,
        'orig_emptied'          : orig_emptied,
        'dest_unchanged'        : dest_unchanged,
        'amount_vs_orig_balance': amount_vs_orig_balance,
        'balance_diff_orig'     : balance_diff_orig,
        'balance_diff_dest'     : balance_diff_dest,
        'amount_mismatch'       : amount_mismatch
    }])

# ================================
# ROUTES
# ================================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    step           = float(data['step'])
    amount         = float(data['amount'])
    oldbalanceOrg  = float(data['oldbalanceOrg'])
    newbalanceOrig = float(data['newbalanceOrig'])
    oldbalanceDest = float(data['oldbalanceDest'])
    newbalanceDest = float(data['newbalanceDest'])
    txn_type       = data['txn_type']

    input_df = preprocess(step, amount, oldbalanceOrg, newbalanceOrig,
                          oldbalanceDest, newbalanceDest, txn_type)

    # ================================
    # MODEL PREDICTIONS
    # ================================
    xgb_prob = float(xgb_model.predict_proba(input_df)[0][1])

    iso_raw        = iso_model.decision_function(input_df)[0]
    iso_train_raw  = iso_model.decision_function(X_test)
    iso_prob       = float(np.clip(
                        1 - (iso_raw - iso_train_raw.min()) /
                        (iso_train_raw.max() - iso_train_raw.min()), 0, 1))

    ensemble_score = (0.6 * xgb_prob) + (0.4 * iso_prob)
    prediction     = "FRAUD" if ensemble_score >= 0.5 else "LEGITIMATE"

    # ================================
    # SHAP EXPLANATION
    # ================================
    shap_values = explainer.shap_values(input_df)

    # Get top 3 important features
    shap_importance = np.abs(shap_values[0])
    feature_names = input_df.columns

    top_indices = np.argsort(shap_importance)[-3:][::-1]

    top_features = []
    for i in top_indices:
        top_features.append({
            "feature": feature_names[i],
            "impact": float(shap_values[0][i])
        })

    # ================================
    # RESPONSE
    # ================================
    return jsonify({
        'xgb_score'      : round(xgb_prob, 4),
        'iso_score'      : round(iso_prob, 4),
        'ensemble_score' : round(ensemble_score, 4),
        'prediction'     : prediction,
        'top_features'   : top_features
    })

# ================================
# RUN
# ================================
if __name__ == '__main__':
    app.run(debug=True)