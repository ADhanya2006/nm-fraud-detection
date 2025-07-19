import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI-EBPL: Fraud Detection", layout="wide", initial_sidebar_state="auto")

# -----------------------------
# CUSTOM STYLING (Violet Gradient + Dark)
# -----------------------------
st.markdown("""
    <style>
    html, body, [class*="css"] {
        background-color: #1e1e2f;
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background: linear-gradient(90deg, #6a0dad, #8a2be2);
        color: white;
        border: none;
        padding: 0.5em 1em;
        border-radius: 10px;
    }
    .stDownloadButton>button {
        background: linear-gradient(90deg, #6a0dad, #8a2be2);
        color: white;
        border: none;
        padding: 0.5em 1em;
        border-radius: 10px;
    }
    .stDataFrame, .stTable {
        background-color: #282846;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.title("🔐 AI-EBPL: Fraud Detection in Financial Transactions")

st.markdown("Upload your financial transaction CSV dataset to begin fraud detection analysis.")

# -----------------------------
# DATA UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("📤 Upload Transaction Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully!")
    st.subheader("📊 Preview of Uploaded Dataset")
    st.dataframe(df.head())

    # -----------------------------
    # DATA PREPROCESSING
    # -----------------------------
    def preprocess_data(df):
        df_clean = df.copy()
        label_cols = ['TransactionType', 'Location', 'DeviceID', 'Channel', 'CustomerOccupation', 'MerchantID']
        for col in label_cols:
            if col in df_clean.columns:
                le = LabelEncoder()
                df_clean[col] = le.fit_transform(df_clean[col])
        return df_clean

    # Simulate fraud column if not present
    if 'is_fraud' not in df.columns:
        np.random.seed(42)
        df['is_fraud'] = np.random.choice([0, 1], size=len(df), p=[0.95, 0.05])

    df_processed = preprocess_data(df)

    # Drop unused columns
    drop_cols = ['TransactionID', 'AccountID', 'TransactionDate', 'PreviousTransactionDate', 'IP Address', 'is_fraud']
    X = df_processed.drop(columns=[col for col in drop_cols if col in df_processed.columns])
    y = df_processed['is_fraud']

    # -----------------------------
    # MODEL TRAINING
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    st.subheader("📈 Model Performance")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(2))

    # -----------------------------
    # FEATURE IMPORTANCE PLOT
    # -----------------------------
    st.subheader("🔥 Feature Importance")
    feature_imp = model.feature_importances_
    features = X.columns
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=feature_imp, y=features, palette="Purples", ax=ax)
    ax.set_title("Top Predictive Features", color='white')
    ax.set_xlabel("Importance", color='white')
    ax.set_ylabel("Feature", color='white')
    fig.patch.set_facecolor('#1e1e2f')
    ax.set_facecolor('#1e1e2f')
    ax.tick_params(colors='white')
    st.pyplot(fig)

    # -----------------------------
    # PREDICTION ON SAME DATASET
    # -----------------------------
    st.subheader("🧪 Predict Fraud on This Dataset")
    fraud_preds = model.predict(X)
    df['Fraud_Prediction'] = np.where(fraud_preds == 1, 'Fraudulent', 'Legit')

    st.dataframe(df[['TransactionID', 'TransactionAmount', 'Fraud_Prediction']].head(20))

    # -----------------------------
    # DOWNLOAD RESULTS
    # -----------------------------
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Results CSV", data=csv, file_name="fraud_detection_results.csv", mime="text/csv")

else:
    st.info("👆 Please upload a CSV file to begin.")
