# Premier League Match Predictor – Backend API

This Flask-based API uses machine learning models trained on recent team form data to predict match outcomes in the English Premier League.

---

## What it predicts:
- Probability of Over 1.5 & Over 2.5 Goals
- Probability of Both Teams To Score (BTTS)
- Match Outcome: Home Win, Draw, Away Win
- Predicted Total Corners

---

## API Endpoint:
POST /predict

Sample Input:
{
  "home_team": "Ipswich",
  "away_team": "Arsenal"
}

Sample Output:
{
  "over_15": "0.78",
  "over_25": "0.41",
  "btts": "0.47",
  "home_win": "0.03",
  "draw": "0.11",
  "away_win": "0.86",
  "corners": 10.02
}

---

## Tech Stack:
- Python (Flask)
- Pandas
- Scikit-learn
- joblib
- Hosted on Render

---

## Getting Started Locally:
1. Clone the repo:
   git clone https://github.com/Ranveersial/premier-league-predictor-backend
2. Install dependencies:
   pip install -r requirements.txt
3. Run the app:
   python app.py

---

## Project Structure:
- app.py
- epl_team_form_features_updated.xlsx
- scaler_model.joblib
- lr_model_over_1_5_calibrated.joblib
- lr_model_over_2_5_calibrated.joblib
- btts_model_calibrated.joblib
- win_model_calibrated.joblib
- corner_model.joblib

---

## Frontend:
See the frontend repo here: https://github.com/Ranveersial/premier-league-predictor-frontend

---

## Author:
Created by Ranveer Singh Sial
GitHub: https://github.com/Ranveersial 
LinkedIn: https://linkedin.com/in/ranveer-sial-444634250/
