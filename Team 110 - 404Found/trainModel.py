import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("loan_approval_dataset.csv")

# Remove spaces from column names
df.columns = df.columns.str.strip()

# Convert text columns
le1 = LabelEncoder()
le2 = LabelEncoder()

df["education"] = le1.fit_transform(df["education"])
df["self_employed"] = le2.fit_transform(df["self_employed"])

# Output column
df["loan_status"] = df["loan_status"].map({
    " Approved":1,
    " Rejected":0,
    "Approved":1,
    "Rejected":0
})

# Features
X = df[[
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "bank_asset_value"
]]

y = df["loan_status"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    random_state=42
)

model.fit(X_train, y_train)

# Save
joblib.dump(model, "loan_model.pkl")

print("loan_model.pkl created successfully")