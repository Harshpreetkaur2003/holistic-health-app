# train_model.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# -----------------------------
# Ensure folders exist
os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)

# Load dataset
data_path = "data/Wellbeing_and_lifestyle_data_Kaggle.csv"
df = pd.read_csv(data_path)

# -----------------------------
# Fill missing values
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Convert numeric columns stored as object
df['DAILY_STRESS'] = pd.to_numeric(df['DAILY_STRESS'], errors='coerce')
df['AGE'] = pd.to_numeric(df['AGE'], errors='coerce')

# -----------------------------
# Encode gender as single column: Male=0, Female=1, Other=2
gender_map = {"Male":0, "Female":1, "Other":2}
df['GENDER_NUM'] = df['GENDER'].map(gender_map)

# -----------------------------
# Target: High risk if stress>3, sleep<6, steps<5000
df['risk_label'] = ((df['DAILY_STRESS']>3) & (df['SLEEP_HOURS']<6) & (df['DAILY_STEPS']<5000)).astype(int)

# -----------------------------
# Features & target
feature_cols = [
    'SLEEP_HOURS','DAILY_STRESS','DAILY_STEPS','WEEKLY_MEDITATION',
    'FRUITS_VEGGIES','TIME_FOR_PASSION','SUPPORTING_OTHERS','SOCIAL_NETWORK',
    'BMI_RANGE','WORK_LIFE_BALANCE_SCORE','AGE','GENDER_NUM'
]

X = df[feature_cols]
y = df['risk_label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Scale numeric columns (first 11 numeric)
numeric_cols = [
    'SLEEP_HOURS','DAILY_STRESS','DAILY_STEPS','WEEKLY_MEDITATION',
    'FRUITS_VEGGIES','TIME_FOR_PASSION','SUPPORTING_OTHERS','SOCIAL_NETWORK',
    'BMI_RANGE','WORK_LIFE_BALANCE_SCORE','AGE'
]
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

# -----------------------------
# RandomForest Model
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("RandomForest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

with open("model/rf_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)
print("RandomForest model saved!")

# -----------------------------
# Neural Network
y_categorical = to_categorical(y_train)
nn_model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(16, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')
])
nn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train_scaled, y_categorical, epochs=50, batch_size=16, verbose=1)

# Evaluate NN
loss, acc = nn_model.evaluate(X_test_scaled, to_categorical(y_test), verbose=0)
print(f"Neural Network Accuracy: {acc:.2f}")

# Save NN model and scaler
nn_model.save("model/nn_model.h5")
with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("Neural Network model and scaler saved!")

