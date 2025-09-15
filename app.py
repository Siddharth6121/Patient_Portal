import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import joblib

# ==============================
# Load tuned Random Forest model
# ==============================
model = joblib.load("models/rf_readmission_classweight.pkl")

# Get feature names from the trained model
FEATURES = model.feature_names_in_

# ==============================
# Helper function to build input row
# ==============================
def build_input(age, num_procedures, days_in_hospital, comorbidity_score, gender):
    # Start with all zeros
    row = pd.DataFrame(np.zeros((1, len(FEATURES))), columns=FEATURES)

    # Fill numeric values
    row.at[0, "age"] = age
    row.at[0, "num_procedures"] = num_procedures
    row.at[0, "days_in_hospital"] = days_in_hospital
    row.at[0, "comorbidity_score"] = comorbidity_score

    # Handle gender encoding
    if "gender_Male" in FEATURES:
        row.at[0, "gender_Male"] = 1 if gender == "Male" else 0
    if "gender_Female" in FEATURES:
        row.at[0, "gender_Female"] = 1 if gender == "Female" else 0

    return row

# ==============================
# Risk Buckets
# ==============================
def risk_bucket(prob):
    if prob >= 0.70:
        return f" High Risk: Patient is likely to be readmitted (Probability: {prob:.2f})"
    elif prob >= 0.40:
        return f" Medium Risk: Patient may need closer monitoring (Probability: {prob:.2f})"
    else:
        return f" Low Risk: Patient is NOT likely to be readmitted (Probability: {prob:.2f})"

# ==============================
# Initialize Dash app
# ==============================
app = dash.Dash(__name__)
app.title = "Patient Readmission Predictor"

app.layout = html.Div([
    html.H1(" Patient Readmission Prediction", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Age"),
        dcc.Input(id="age", type="number", value=50, min=0, max=120, step=1),

        html.Label("Number of Procedures"),
        dcc.Input(id="num_procedures", type="number", value=2, min=0, max=20, step=1),

        html.Label("Days in Hospital"),
        dcc.Input(id="days_in_hospital", type="number", value=5, min=1, max=60, step=1),

        html.Label("Comorbidity Score"),
        dcc.Input(id="comorbidity_score", type="number", value=1, min=0, max=10, step=1),

        html.Label("Gender"),
        dcc.Dropdown(
            id="gender",
            options=[{"label": "Male", "value": "Male"}, {"label": "Female", "value": "Female"}],
            value="Male"
        ),

        html.Br(),
        html.Button("Predict Readmission", id="predict-btn", n_clicks=0,
                    style={"backgroundColor": "#4CAF50", "color": "white", "padding": "10px"}),

        html.Div(id="prediction-output", style={"marginTop": "20px", "fontSize": "20px"})
    ], style={"width": "50%", "margin": "auto"})
])

# ==============================
# Callbacks
# ==============================
@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    State("age", "value"),
    State("num_procedures", "value"),
    State("days_in_hospital", "value"),
    State("comorbidity_score", "value"),
    State("gender", "value")
)
def predict(n_clicks, age, num_procedures, days_in_hospital, comorbidity_score, gender):
    if n_clicks > 0:
        # Build input vector
        X_new = build_input(age, num_procedures, days_in_hospital, comorbidity_score, gender)

        # Debug print to terminal
        print("Model expects:", list(FEATURES))
        print("X_new columns:", list(X_new.columns))
        print("X_new row:", X_new.to_dict(orient="records")[0])

        # Get probability
        prob = model.predict_proba(X_new)[0][1]

        # Get bucketed message
        return risk_bucket(prob)

    return ""

# ==============================
# Run App
# ==============================
if __name__ == "__main__":
    app.run(debug=True)
