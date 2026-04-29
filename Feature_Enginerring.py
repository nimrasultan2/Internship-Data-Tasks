import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures

# =============================================================================
# 1. CREATE DATASET  (same as before)
# =============================================================================
np.random.seed(42)
n = 1000

df = pd.DataFrame({
    "Gender":            np.random.choice([0, 1], n),
    "Married":           np.random.choice([0, 1], n),
    "Dependents":        np.random.choice([0, 1, 2, 3], n),
    "Education":         np.random.choice([0, 1], n),
    "Self_Employed":     np.random.choice([0, 1], n),
    "ApplicantIncome":   np.random.randint(1500, 15000, n),
    "CoapplicantIncome": np.random.randint(0, 7000, n),
    "LoanAmount":        np.random.randint(50, 700, n),
    "Loan_Amount_Term":  np.random.choice([120, 180, 240, 360], n),
    "Credit_History":    np.random.choice([0, 1], n, p=[0.2, 0.8]),
    "Property_Area":     np.random.choice([0, 1, 2], n),
})

score = (
    (df["Credit_History"] == 1).astype(int) * 3 +
    (df["ApplicantIncome"] > 5000).astype(int) * 2 +
    (df["Education"] == 1).astype(int) +
    (df["LoanAmount"] < 300).astype(int)
)
df["Loan_Status"] = (score >= 4).astype(int)

# =============================================================================
# 2. BASELINE MODEL  (no feature engineering)
# =============================================================================
X_base = df.drop("Loan_Status", axis=1)
y      = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(X_base, y, test_size=0.2, random_state=42)

base_model = RandomForestClassifier(n_estimators=100, random_state=42)
base_model.fit(X_train, y_train)
base_acc = accuracy_score(y_test, base_model.predict(X_test))
print(f"Baseline Accuracy : {base_acc * 100:.2f}%")

# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================
fe = df.drop("Loan_Status", axis=1).copy()

# --- 3a. Interaction features ---------------------------------------------
fe["Total_Income"]        = fe["ApplicantIncome"] + fe["CoapplicantIncome"]
fe["Income_x_Credit"]     = fe["Total_Income"] * fe["Credit_History"]
fe["Married_x_Income"]    = fe["Married"] * fe["ApplicantIncome"]

# --- 3b. Polynomial features (degree=2 on numeric columns) ---------------
num_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_array = poly.fit_transform(fe[num_cols])
poly_names = poly.get_feature_names_out(num_cols)
poly_df    = pd.DataFrame(poly_array, columns=poly_names)

# Drop the original columns from poly_df to avoid duplicates, then concat
poly_df = poly_df.drop(columns=num_cols)
fe = pd.concat([fe.reset_index(drop=True), poly_df.reset_index(drop=True)], axis=1)

# --- 3c. Binning / categorization ----------------------------------------
fe["Income_Band"]  = pd.cut(fe["ApplicantIncome"],
                             bins=[0, 3000, 6000, 10000, 99999],
                             labels=[0, 1, 2, 3]).astype(int)

fe["Loan_Band"]    = pd.cut(fe["LoanAmount"],
                             bins=[0, 150, 300, 500, 99999],
                             labels=[0, 1, 2, 3]).astype(int)

fe["DTI_Ratio"]    = fe["LoanAmount"] / (fe["Total_Income"] + 1)  # debt-to-income

print(f"\nFeatures before engineering : {X_base.shape[1]}")
print(f"Features after  engineering : {fe.shape[1]}")

# =============================================================================
# 4. FEATURE SELECTION
# =============================================================================
X_fe_train, X_fe_test, y_train, y_test = train_test_split(fe, y, test_size=0.2, random_state=42)

# --- 4a. Feature importance from Random Forest ---------------------------
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selector.fit(X_fe_train, y_train)

importance_df = pd.DataFrame({
    "Feature":    fe.columns,
    "Importance": rf_selector.feature_importances_
}).sort_values("Importance", ascending=False)

print("\n--- Feature Importance (top 10) ---")
print(importance_df.head(10).to_string(index=False))

# Keep top 10 features by importance
top10_features = importance_df.head(10)["Feature"].tolist()

# --- 4b. Correlation analysis --------------------------------------------
corr = fe.corrwith(y).abs().sort_values(ascending=False)
print("\n--- Correlation with Target (top 10) ---")
print(corr.head(10))

top_corr_features = corr.head(10).index.tolist()

# --- 4c. Recursive Feature Elimination (RFE) -----------------------------
rfe = RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=42),
          n_features_to_select=10)
rfe.fit(X_fe_train, y_train)
rfe_features = fe.columns[rfe.support_].tolist()

print("\n--- RFE Selected Features ---")
print(rfe_features)

# --- Combine all selected features (union) --------------------------------
selected = list(set(top10_features + top_corr_features + rfe_features))
print(f"\nFinal selected feature count : {len(selected)}")

# =============================================================================
# 5. TRAIN FINAL MODEL ON ENGINEERED + SELECTED FEATURES
# =============================================================================
X_final_train = X_fe_train[selected]
X_final_test  = X_fe_test[selected]

final_model = RandomForestClassifier(n_estimators=100, random_state=42)
final_model.fit(X_final_train, y_train)
final_acc = accuracy_score(y_test, final_model.predict(X_final_test))
print(f"\nFinal Model Accuracy : {final_acc * 100:.2f}%")

# =============================================================================
# 6. BEFORE vs AFTER COMPARISON
# =============================================================================
print("\n========================================")
print("        PERFORMANCE COMPARISON          ")
print("========================================")
print(f"  Baseline (11 features)  : {base_acc * 100:.2f}%")
print(f"  After FE ({len(selected):2d} features) : {final_acc * 100:.2f}%")
improvement = (final_acc - base_acc) * 100
print(f"  Improvement             : {improvement:+.2f}%")
print("========================================")