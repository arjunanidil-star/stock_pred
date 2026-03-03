%%writefile app.py
import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide")
st.title("📈 Same-Day Stock Close Prediction (Alpha Vantage)")

st.sidebar.header("User Input")

api_key = st.sidebar.text_input("Alpha Vantage API Key", type="password")
stock = st.sidebar.text_input("Stock Symbol", "AAPL")

if st.sidebar.button("Run Prediction"):

    if not api_key:
        st.error("Please enter your Alpha Vantage API key.")
        st.stop()

    st.info("Fetching data from Alpha Vantage...")

    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={stock}&outputsize=compact&apikey={api_key}"
    response = requests.get(url)
    data_json = response.json()

    if "Time Series (Daily)" not in data_json:
        st.error(f"API Error: {data_json}")
        st.stop()

    df = pd.DataFrame.from_dict(
        data_json["Time Series (Daily)"],
        orient="index"
    )

    df = df.rename(columns={
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close",
        "5. volume": "Volume"
    })

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.astype(float)

    df["ROC"] = df["Close"].pct_change(periods=14) * 100
    df.dropna(inplace=True)

    if df.empty:
        st.error("Not enough data after ROC calculation.")
        st.stop()

    # ===============================
    # Train Model Dynamically
    # ===============================
    X = df[["Open", "High", "Low", "ROC"]]
    y = df["Close"]

    model = RandomForestRegressor(max_depth=3, random_state=0)
    model.fit(X, y)

    df["Predicted_Close"] = model.predict(X)

    st.subheader("Prediction Table")
    st.dataframe(df.tail())

    st.subheader("Actual vs Predicted Close")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["Close"], label="Actual Close")
    ax.plot(df.index, df["Predicted_Close"], label="Predicted Close")
    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")

    st.pyplot(fig)
