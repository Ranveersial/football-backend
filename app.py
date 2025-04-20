from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

# Initialize app
app = Flask(__name__)
CORS(app)

# Load models and data
lr_15 = joblib.load('lr_model_over_1_5.joblib')
lr_25 = joblib.load('lr_model_over_2_5.joblib')
btts_model = joblib.load('btts_model.joblib')
win_model = joblib.load('win_model.joblib')
corner_model = joblib.load('corner_model.joblib')
scaler = joblib.load('scaler_model.joblib')
epl_data = pd.read_excel('epl_team_form_features_updated.xlsx')

# Feature extraction
def get_form_features(data, home_team, away_team, n=10):
    def compute(team, role):
        if role == 'Home':
            matches = data[data['HomeTeam'] == team].sort_values(by='Date', ascending=False).head(n)
            return pd.Series({
                'WinRate': matches['HomeWin'].mean(),
                'DrawRate': matches['Draw'].mean(),
                'LossRate': 1 - (matches['HomeWin'] + matches['Draw']).mean(),
                'CleanSheetRate': matches['HomeCleanSheet'].mean(),
                'BTTSRate': matches['BTTS'].mean(),
                'AvgGoalDiff': matches['HomeGoalDiff'].mean(),
                'AvgTotalGoals': matches['TotalGoalsHT'].mean(),
                'AvgShots': matches['HS'].mean(),
                'AvgShotsOnTarget': matches['HST'].mean(),
                'AvgCorners': matches['HC'].mean(),
                'AvgCardsY': matches['HY'].mean(),
                'AvgCardsR': matches['HR'].mean()
            })
        else:
            matches = data[data['AwayTeam'] == team].sort_values(by='Date', ascending=False).head(n)
            return pd.Series({
                'WinRate': matches['AwayWin'].mean(),
                'DrawRate': matches['Draw'].mean(),
                'LossRate': 1 - (matches['AwayWin'] + matches['Draw']).mean(),
                'CleanSheetRate': matches['AwayCleanSheet'].mean(),
                'BTTSRate': matches['BTTS'].mean(),
                'AvgGoalDiff': matches['AwayGoalDiff'].mean(),
                'AvgTotalGoals': matches['TotalGoalsHT'].mean(),
                'AvgShots': matches['AS'].mean(),
                'AvgShotsOnTarget': matches['AST'].mean(),
                'AvgCorners': matches['AC'].mean(),
                'AvgCardsY': matches['AY'].mean(),
                'AvgCardsR': matches['AR'].mean()
            })
    home_form = compute(home_team, 'Home').add_prefix('Home_')
    away_form = compute(away_team, 'Away').add_prefix('Away_')
    f = pd.concat([home_form, away_form]).to_frame().T.fillna(0)
    return scaler.transform(f)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data_json = request.get_json()
    home = data_json['home_team']
    away = data_json['away_team']
    
    X = get_form_features(epl_data, home, away)

    # Predict
    over15 = lr_15.predict_proba(X)[0][1]
    over25 = lr_25.predict_proba(X)[0][1]
    btts = btts_model.predict_proba(X)[0][1]
    win_probs = win_model.predict_proba(X)[0]
    corners = corner_model.predict(X)[0]

    return jsonify({
        "over_15": f"{over15:.2f}",
        "over_25": f"{over25:.2f}",
        "btts": f"{btts:.2f}",
        "home_win": f"{win_probs[0]:.2f}",
        "draw": f"{win_probs[1]:.2f}",
        "away_win": f"{win_probs[2]:.2f}",
        "corners": round(corners, 2)
    })

import os

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))

