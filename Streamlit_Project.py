import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# ── Create a simple loan dataset ──────────────────────────────────────────────
np.random.seed(42)
n = 1000

gender              = np.random.choice([0, 1], n)               # 0=Female, 1=Male
married             = np.random.choice([0, 1], n)               # 0=No, 1=Yes
dependents          = np.random.choice([0, 1, 2, 3], n)
education           = np.random.choice([0, 1], n)               # 0=Not Graduate, 1=Graduate
self_employed       = np.random.choice([0, 1], n)               # 0=No, 1=Yes
applicant_income    = np.random.randint(1500, 15000, n)
coapplicant_income  = np.random.randint(0, 7000, n)
loan_amount         = np.random.randint(50, 700, n)             # in thousands
loan_term           = np.random.choice([120, 180, 240, 360], n) # in months
credit_history      = np.random.choice([0, 1], n, p=[0.2, 0.8])# 0=Bad, 1=Good
property_area       = np.random.choice([0, 1, 2], n)            # 0=Rural,1=Semiurban,2=Urban

# Simple rule-based target so the model has something real to learn
score = (
    (credit_history == 1).astype(int) * 3 +
    (applicant_income > 5000).astype(int) * 2 +
    (education == 1).astype(int) +
    (loan_amount < 300).astype(int)
)
loan_status = (score >= 4).astype(int)   # 1=Approved, 0=Rejected

df = pd.DataFrame({
    "Gender":            gender,
    "Married":           married,
    "Dependents":        dependents,
    "Education":         education,
    "Self_Employed":     self_employed,
    "ApplicantIncome":   applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount":        loan_amount,
    "Loan_Amount_Term":  loan_term,
    "Credit_History":    credit_history,
    "Property_Area":     property_area,
    "Loan_Status":       loan_status
})

print("Dataset shape:", df.shape)
print("Approval rate: {:.1f}%".format(df["Loan_Status"].mean() * 100))

# ── Features & target ─────────────────────────────────────────────────────────
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Train Random Forest ───────────────────────────────────────────────────────
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Model Accuracy: {acc * 100:.2f}%")

# ── Save model ────────────────────────────────────────────────────────────────
with open("loan_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as loan_model.pkl")