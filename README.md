# 🏥 Patient Portal – Readmission Prediction

## 📌 Project Overview
This project builds a **machine learning model** to predict whether a patient is likely to be readmitted to the hospital within **30 days of discharge**.  
It is designed as a **proof of concept** for clinical decision support and future integration into a patient portal system.

---

## 📂 Repository Structure
Patient_Portal/
│
├── data/ # Datasets (train, test, sample submission)
├── notebooks/ # Jupyter notebooks for model experimentation
├── models/ # (Not pushed to GitHub, stored externally)
├── app.py # Dash web application for interactive predictions
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── Patient_Readmission_Final_Report.pdf (external, see below)


---

## 📊 Dataset

- Source: [Kaggle – Hospital Readmission Dataset by vanpatangan](https://www.kaggle.com/datasets/vanpatangan/readmission-dataset) :contentReference[oaicite:0]{index=0}  
- Structure: Includes **train_df.csv**, **test_df.csv**, and **sample_submission.csv**. :contentReference[oaicite:1]{index=1}  
- Features used: age, number of procedures, days in hospital, comorbidity score, gender, etc. :contentReference[oaicite:2]{index=2}  
- Target: `readmitted` (1 = patient readmitted within 30 days, 0 = not readmitted) :contentReference[oaicite:3]{index=3}  

⚠️ Note: The dataset is **imbalanced** (your local data had ~812 “not readmitted” vs 188 “readmitted”).  
This imbalance created a major challenge for model performance because models tended to predict the majority class.  


---

## 🧠 Modeling Journey
We experimented with multiple approaches:
1. **Baseline Logistic Regression** – Simple, interpretable, but poor recall.  
2. **Random Forest Classifier** – Balanced performance, interpretable feature importances.  
   - Tuned with GridSearch  
   - Tried **SMOTE oversampling**, **class weights**, and **probability calibration**.  
3. **XGBoost** – Tested, but performed worse on recall (dropped).  

📌 Final Choice: **Random Forest Classifier (tuned)**  
- Best trade-off between recall and interpretability.  
- Still limited due to dataset imbalance.  

---

## 📈 Results Summary
| Model Variant             | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|----------------------------|----------|-----------|--------|----------|---------|
| RF + SMOTE (tuned)        | 0.71     | 0.17      | 0.14   | 0.15     | 0.51    |
| RF + Calibration          | 0.81     | 0.00      | 0.00   | 0.00     | 0.51    |
| RF + Class Weights        | 0.80     | 0.00      | 0.00   | 0.00     | 0.52    |

---

## 🖥️ App Demo
We built an **interactive web app** using **Dash**.  
- Enter patient details → Get **risk prediction (Low, Medium, High)**.  
- Example input:  
  - Age = 65  
  - Procedures = 3  
  - Days in hospital = 7  
  - Comorbidity Score = 2  
  - Gender = Male  
  → **Prediction: High Risk (Probability 0.94)**  

Run locally:
```bash
pip install -r requirements.txt
python app.py


📦 Requirements

Python 3.9+

pandas, numpy, scikit-learn, imbalanced-learn

xgboost, dash, plotly

Install all dependencies:


pip install -r requirements.txt

📑 Project Report

The detailed project summary, methodology, results, and future improvements are documented in:
📄 Patient_Readmission_Final_Report.pdf

⚠️ Not included in repo due to size → Download here


🤖 Model Files

The trained models (.pkl) are stored externally due to GitHub file size limits.
Download from: Google Drive Link


🚀 Next Steps

Acquire a larger, more balanced dataset for training.

Explore advanced models (LightGBM, CatBoost, ensemble stacking).

Deploy app with FastAPI/Streamlit + Docker for real-world use.
