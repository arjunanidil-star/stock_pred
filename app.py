import streamlit as st
import yfinance as yf
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📈 Same-Day Stock Close Prediction")

# Load trained model
model = joblib.load("AAPL_trained_model.pkl")

# Sidebar inputs
st.sidebar.header("User Input")

stock = st.sidebar.text_input("Stock Symbol", "AAPL")
start_date = st.sidebar.date_input("Start Date")
end_date = st.sidebar.date_input("End Date")

if st.sidebar.button("Run Prediction"):

    # Download data
    df = yf.download(stock, start=start_date, end=end_date, progress=False)

    if df.empty:
        st.error("No data found. Check stock symbol or date range.")
    else:
        df["ROC"] = df["Close"].pct_change(periods=14) * 100
        df.dropna(inplace=True)

        if df.empty:
            st.error("Not enough data to compute ROC.")
        else:
            X = df[["Open", "High", "Low", "ROC"]]
            df["Predicted_Close"] = model.predict(X)

            st.subheader("Prediction Table")
            st.dataframe(df.tail())

            # Plot
            st.subheader("Actual vs Predicted Close")

            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(df.index, df["Close"], label="Actual Close")
            ax.plot(df.index, df["Predicted_Close"], label="Predicted Close")
            ax.legend()
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")

            st.pyplot(fig)
