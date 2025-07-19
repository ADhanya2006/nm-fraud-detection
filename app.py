import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("🛡️ AI-EBPL: Fraud Detection in Financial Transactions")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Fraud Detection in Financial Transactions - DataSet.csv")
    return df

df = load_data()

st.subheader("Sample Dataset")
st.write(df.head())

# Encode categorical columns
def preprocess_data(df):
    df_clean = df.copy()
    label_cols = ['TransactionType', 'Location', 'DeviceID', 'Channel', 'CustomerOccupation', 'MerchantID']
    for col in label_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col])
    return df_clean

# Add fraud label manually for this example (in real cases, it's from your dataset)
if 'is_fraud' not in df.columns:
    np.random.seed(42)
    df['is_fraud'] = np.random.choice([0, 1], size=len(df), p=[0.95, 0.05])  # Simulated labels

# Preprocess
df_processed = preprocess_data(df)

# Train-test split
X = df_processed.drop(columns=['TransactionID', 'AccountID', 'TransactionDate',
                               'PreviousTransactionDate', 'IP Address', 'is_fraud'])
y = df_processed['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

st.subheader("Model Summary")
st.text(classification_report(y_test, model.predict(X_test)))

# Feature importance plot
st.subheader("🔍 Feature Importance")
importance = model.feature_importances_
features = X.columns
plt.figure(figsize=(10, 4))
sns.barplot(x=importance, y=features)
plt.title("Top Features Used for Fraud Detection")
st.pyplot(plt)

# Upload new transaction data
st.subheader("📤 Upload New Transaction CSV for Prediction")

uploaded_file = st.file_uploader("Upload a CSV file with the same structure as the original dataset.", type=["csv"])

if uploaded_file is not None:
    new_data = pd.read_csv(uploaded_file)
    new_data_clean = preprocess_data(new_data)

    # Drop unused columns to match training features
    X_new = new_data_clean[X.columns]

    # Predict
    predictions = model.predict(X_new)
    new_data['Fraud_Prediction'] = np.where(predictions == 1, 'Fraudulent', 'Legit')

    st.success("Predictions Completed!")
    st.dataframe(new_data[['TransactionID', 'TransactionAmount', 'Fraud_Prediction']])

    csv = new_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Results CSV", data=csv, file_name="fraud_predictions.csv", mime="text/csv")
