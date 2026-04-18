import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from imblearn.over_sampling import SMOTE

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip()

print("Dataset Loaded ✅")
print(df.head())

# -----------------------------
# 2. Date Processing
# -----------------------------
df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])

df['Transaction Day'] = df['Transaction Date'].dt.day
df['Transaction Month'] = df['Transaction Date'].dt.month

df.drop('Transaction Date', axis=1, inplace=True)

# -----------------------------
# 3. Drop Unnecessary Columns
# -----------------------------
df.drop(['Transaction ID', 'Customer ID', 'IP Address'], axis=1, inplace=True)

# -----------------------------
# 4. Encode Categorical Columns
# -----------------------------
categorical_cols = [
    'Payment Method',
    'Product Category',
    'Customer Location',
    'Device Used',
    'Shipping Address',
    'Billing Address'
]

encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Save encoders
joblib.dump(encoders, "encoders.pkl")

# -----------------------------
# 5. Split Features & Target
# -----------------------------
X = df.drop("Is Fraudulent", axis=1)
y = df["Is Fraudulent"]

print("\nOriginal Class Distribution:")
print(y.value_counts())

# -----------------------------
# 6. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 7. Apply SMOTE (IMPORTANT)
# -----------------------------
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print(np.bincount(y_train))

# -----------------------------
# 8. Feature Scaling
# -----------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# -----------------------------
# 9. Train Model
# -----------------------------
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=10,  # VERY IMPORTANT (handles imbalance)
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# -----------------------------
# 10. Evaluation
# -----------------------------
y_pred = model.predict(X_test)

print("\nModel Evaluation:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# 11. Save Model
# -----------------------------
joblib.dump(model, "model.pkl")

print("\n✅ FINAL MODEL SAVED SUCCESSFULLY!")