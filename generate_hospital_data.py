import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# ==========================================
# Tehran Heart Hospital - Synthetic Dataset
# ==========================================
# IMPORTANT:
# All data in this project are completely
# fictional and created only for education/testing.
# ==========================================

random.seed(42)
np.random.seed(42)

N = 500

patient_ids = [f"THH-{i:04d}" for i in range(1, N + 1)]

first_names = [
    "علی", "محمد", "رضا", "حسین", "امیر",
    "مریم", "سارا", "زهرا", "نگار", "فاطمه",
    "نوید", "آرمان", "میلاد", "سمیه", "لیلا"
]

last_names = [
    "احمدی", "محمدی", "حسینی", "رضایی",
    "کریمی", "مرادی", "موسوی", "کاظمی",
    "نوری", "صادقی", "رستمی", "اکبری"
]

def random_birth_date():
    start = datetime(1940, 1, 1)
    end = datetime(2005, 12, 31)
    days = random.randint(0, (end - start).days)
    return (start + timedelta(days=days)).date()


# ==========================================
# 1 - Patients
# ==========================================

birth_dates = [random_birth_date() for _ in range(N)]

patients = pd.DataFrame({
    "Patient_ID": patient_ids,
    "First_Name": [random.choice(first_names) for _ in range(N)],
    "Last_Name": [random.choice(last_names) for _ in range(N)],
    "Gender": np.random.choice(
        ["Male", "Female"], N, p=[0.56, 0.44]
    ),
    "Birth_Date": birth_dates,
    "Age": [
        int((datetime(2026, 9, 16).date() - d).days / 365.25)
        for d in birth_dates
    ],
    "Department": np.random.choice(
        [
            "Emergency Cardiology",
            "CCU",
            "Cardiac ICU",
            "Angiography",
            "Cardiac Surgery",
            "Cardiology Clinic"
        ],
        N
    )
})


# ==========================================
# 2 - Vital Signs
# ==========================================

vitals = pd.DataFrame({
    "Patient_ID": patient_ids,
    "Systolic_BP": np.clip(
        np.random.normal(132, 20, N).round(),
        85, 210
    ).astype(int),

    "Diastolic_BP": np.clip(
        np.random.normal(82, 12, N).round(),
        50, 130
    ).astype(int),

    "Heart_Rate": np.clip(
        np.random.normal(78, 14, N).round(),
        45, 145
    ).astype(int),

    "Oxygen_Saturation": np.clip(
        np.random.normal(96.5, 2, N).round(1),
        88, 100
    ),

    "Temperature": np.clip(
        np.random.normal(36.8, 0.45, N).round(1),
        35.5, 39.5
    )
})


# ==========================================
# 3 - Laboratory
# ==========================================

laboratory = pd.DataFrame({
    "Patient_ID": patient_ids,

    "Total_Cholesterol": np.clip(
        np.random.normal(195, 42, N).round(),
        90, 380
    ).astype(int),

    "LDL": np.clip(
        np.random.normal(118, 35, N).round(),
        35, 260
    ).astype(int),

    "HDL": np.clip(
        np.random.normal(48, 12, N).round(),
        20, 95
    ).astype(int),

    "Triglycerides": np.clip(
        np.random.normal(155, 75, N).round(),
        40, 500
    ).astype(int),

    "Fasting_Glucose": np.clip(
        np.random.normal(108, 30, N).round(),
        65, 260
    ).astype(int),

    "Creatinine": np.clip(
        np.random.normal(1.0, 0.25, N).round(2),
        0.5, 2.8
    ),

    "Troponin": np.clip(
        np.random.lognormal(-1.7, 1, N).round(3),
        0.01, 20
    )
})


# ==========================================
# 4 - Diagnosis
# ==========================================

diagnoses = [
    "Coronary Artery Disease",
    "Hypertension",
    "Arrhythmia",
    "Heart Failure",
    "Chest Pain",
    "Myocardial Infarction",
    "Valvular Disease",
    "Diabetes with Cardiovascular Disease"
]

diagnosis = pd.DataFrame({
    "Patient_ID": patient_ids,

    "Primary_Diagnosis": np.random.choice(
        diagnoses, N
    ),

    "Severity": np.random.choice(
        ["Mild", "Moderate", "Severe"],
        N,
        p=[0.35, 0.45, 0.20]
    ),

    "Previous_Heart_Attack": np.random.choice(
        ["No", "Yes"], N, p=[0.82, 0.18]
    ),

    "Hypertension_History": np.random.choice(
        ["No", "Yes"], N, p=[0.40, 0.60]
    ),

    "Diabetes_History": np.random.choice(
        ["No", "Yes"], N, p=[0.72, 0.28]
    ),

    "Smoking_History": np.random.choice(
        ["No", "Yes"], N, p=[0.68, 0.32]
    )
})


# ==========================================
# 5 - Medications
# ==========================================

medications_list = [
    "Aspirin",
    "Atorvastatin",
    "Metoprolol",
    "Losartan",
    "Amlodipine",
    "Clopidogrel",
    "Furosemide",
    "Warfarin"
]

medications = pd.DataFrame({
    "Patient_ID": patient_ids,

    "Main_Medication": np.random.choice(
        medications_list, N
    ),

    "Daily_Dose": np.random.choice(
        ["1 tablet", "2 tablets", "3 tablets"],
        N,
        p=[0.45, 0.45, 0.10]
    ),

    "Medication_Status": np.random.choice(
        ["Active", "Temporary", "Discontinued"],
        N,
        p=[0.78, 0.12, 0.10]
    )
})


# ==========================================
# 6 - Diagnostic Procedures
# ==========================================

procedures = pd.DataFrame({
    "Patient_ID": patient_ids,

    "Echocardiography": np.random.choice(
        ["Completed", "Not Completed"],
        N,
        p=[0.65, 0.35]
    ),

    "ECG": np.random.choice(
        ["Completed", "Not Completed"],
        N,
        p=[0.80, 0.20]
    ),

    "Angiography": np.random.choice(
        ["Completed", "Not Completed"],
        N,
        p=[0.30, 0.70]
    ),

    "Ejection_Fraction": np.clip(
        np.random.normal(55, 10, N).round(1),
        20, 75
    )
})


# ==========================================
# 7 - Hospital Admission
# ==========================================

admission_dates = [
    datetime(2025, 1, 1) +
    timedelta(days=random.randint(0, 620))
    for _ in range(N)
]

discharge_dates = [
    d + timedelta(days=random.randint(1, 15))
    for d in admission_dates
]

admission = pd.DataFrame({
    "Patient_ID": patient_ids,

    "Admission_Date": [
        d.date() for d in admission_dates
    ],

    "Discharge_Date": [
        d.date() for d in discharge_dates
    ],

    "Hospital_Days": [
        (d - a).days
        for a, d in zip(
            admission_dates,
            discharge_dates
        )
    ],

    "Admission_Type": np.random.choice(
        ["Emergency", "Elective"],
        N,
        p=[0.62, 0.38]
    ),

    "Discharge_Status": np.random.choice(
        [
            "Improved",
            "Transferred",
            "Outpatient Follow-up"
        ],
        N,
        p=[0.78, 0.08, 0.14]
    )
})


# ==========================================
# 8 - Surgery
# ==========================================

surgery_types = [
    "No Surgery",
    "CABG",
    "Valve Surgery",
    "Other"
]

surgery = pd.DataFrame({
    "Patient_ID": patient_ids,

    "Surgery_Type": np.random.choice(
        surgery_types,
        N,
        p=[0.72, 0.16, 0.07, 0.05]
    ),

    "Surgery_Date": [
        (
            d + timedelta(
                days=random.randint(1, 12)
            )
        ).date()
        if random.random() < 0.28
        else None
        for d in admission_dates
    ],

    "Surgery_Status": np.random.choice(
        [
            "Not Applicable",
            "Successful",
            "Needs Follow-up"
        ],
        N,
        p=[0.72, 0.23, 0.05]
    )
})


# ==========================================
# 9 - Follow-up
# ==========================================

followup = pd.DataFrame({
    "Patient_ID": patient_ids,

    "Followup_Date": [
        (
            d + timedelta(
                days=random.randint(14, 120)
            )
        ).date()
        for d in admission_dates
    ],

    "Symptom_Status": np.random.choice(
        [
            "Improved",
            "No Change",
            "Needs Evaluation"
        ],
        N,
        p=[0.60, 0.28, 0.12]
    ),

    "Followup_BP": np.clip(
        np.random.normal(128, 17, N).round(),
        90, 190
    ).astype(int),

    "Medication_Adherence": np.random.choice(
        ["Good", "Moderate", "Poor"],
        N,
        p=[0.62, 0.28, 0.10]
    )
})


# ==========================================
# 10 - Clinical Summary
# ==========================================

risk_value = np.clip(
    (vitals["Systolic_BP"] - 120) / 120
    + (laboratory["LDL"] - 100) / 300
    + (diagnosis["Smoking_History"] == "Yes") * 0.2
    + (diagnosis["Diabetes_History"] == "Yes") * 0.2
    + (diagnosis["Previous_Heart_Attack"] == "Yes") * 0.3,
    0,
    1
)

clinical_summary = pd.DataFrame({
    "Patient_ID": patient_ids,

    "Synthetic_Risk_Group": pd.cut(
        risk_value,
        bins=[-0.01, 0.30, 0.60, 1.01],
        labels=["Low", "Moderate", "High"]
    ),

    "Followup_Required": np.where(
        risk_value > 0.55,
        "Yes",
        "No"
    ),

    "Data_Note": [
        "Synthetic data - educational/testing only"
    ] * N
})


# ==========================================
# Create Excel workbook
# ==========================================

sheets = {
    "Patients": patients,
    "Vital_Signs": vitals,
    "Laboratory": laboratory,
    "Diagnosis": diagnosis,
    "Medications": medications,
    "Diagnostic_Procedures": procedures,
    "Admission": admission,
    "Surgery": surgery,
    "Followup": followup,
    "Clinical_Summary": clinical_summary
}

output_file = "tehran_heart_hospital_500.xlsx"

with pd.ExcelWriter(
    output_file,
    engine="openpyxl"
) as writer:

    for sheet_name, dataframe in sheets.items():
        dataframe.to_excel(
            writer,
            sheet_name=sheet_name,
            index=False
        )

print("=" * 60)
print("Excel file created successfully!")
print("=" * 60)
print(f"File: {output_file}")
print(f"Patients: {N}")
print(f"Sheets: {len(sheets)}")
print("All data are SYNTHETIC / FICTIONAL.")
print("=" * 60)
