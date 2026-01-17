import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

# -------------------- PAGE CONFIG --------------------
st.set_page_config(layout="wide")
st.title("📈 Cryptocurrency Market Forecasting App")

tab1, tab2 = st.tabs(["📄 Project Summary", "📊 Forecasting App"])

# ====================================================
# TAB 1 — SUMMARY
# ====================================================
with tab1:
    st.markdown("""
    ### 🔍 Objective
    Forecast cryptocurrency market behavior using historical data.

    ### 📦 Dataset
    - Time-series crypto market data
    - Metrics: price, market cap, trading volume

    ### 🧠 Models Used
    - ARIMA
    - SARIMA
    - Exponential Smoothing
    - ARCH / GARCH (volatility)
    - LSTM Neural Network

    ### 📈 Output
    - 90-day forecast
    - Trend & seasonality analysis
    - Volatility modeling
    """)

# ====================================================
# TAB 2 — FORECASTING
# ====================================================
with tab2:
    uploaded_file = st.sidebar.file_uploader("Upload crypto-markets.csv", type=["csv"])

    if uploaded_file is None:
        st.warning("Upload the crypto dataset to proceed.")
        st.stop()

    # -------------------- LOAD DATA --------------------
    df = pd.read_csv(uploaded_file)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.sort_values("date", inplace=True)

    # -------------------- SIDEBAR FILTERS --------------------
    st.sidebar.subheader("Filters")
    symbol = st.sidebar.selectbox("Cryptocurrency", df["symbol"].unique())
    target_col = st.sidebar.selectbox(
        "Target Variable",
        ["price", "market_cap", "volume"]
    )

    filtered_df = df[df["symbol"] == symbol].copy()
    filtered_df.set_index("date", inplace=True)

    series = filtered_df[target_col].dropna()

    # -------------------- DATA PREVIEW --------------------
    st.subheader(f"{symbol} — {target_col.upper()} Preview")
    st.write(series.head())

    # -------------------- TIME SERIES PLOT --------------------
    fig = px.line(series, title=f"{symbol} {target_col.upper()} Over Time")
    st.plotly_chart(fig, use_container_width=True)

    # -------------------- DECOMPOSITION --------------------
    st.subheader("Seasonal Decomposition")
    decomposition = sm.tsa.seasonal_decompose(series, model="additive", period=30)

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True)
    fig.add_trace(go.Scatter(y=decomposition.observed), 1, 1)
    fig.add_trace(go.Scatter(y=decomposition.trend), 2, 1)
    fig.add_trace(go.Scatter(y=decomposition.seasonal), 3, 1)
    fig.add_trace(go.Scatter(y=decomposition.resid), 4, 1)
    fig.update_layout(height=800)
    st.plotly_chart(fig, use_container_width=True)

    # -------------------- ADF TEST --------------------
    st.subheader("ADF Stationarity Test")
    adf_result = adfuller(series.dropna())
    st.write({
        "ADF Statistic": adf_result[0],
        "p-value": adf_result[1],
        "Lags": adf_result[2]
    })

    # -------------------- TRAIN / TEST --------------------
    train = series[:-90]
    test = series[-90:]

    # ====================================================
    # ARIMA
    # ====================================================
    if st.button("Run ARIMA Forecast"):
        model = ARIMA(train, order=(5,1,0)).fit()
        forecast = model.forecast(90)

        rmse = np.sqrt(mean_squared_error(test, forecast))
        r2 = r2_score(test, forecast)

        st.success("ARIMA Metrics")
        st.write(f"RMSE: {rmse:.2f}")
        st.write(f"R²: {r2:.4f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=train, name="Train"))
        fig.add_trace(go.Scatter(y=test, name="Test"))
        fig.add_trace(go.Scatter(y=forecast, name="Forecast"))
        st.plotly_chart(fig, use_container_width=True)

    # ====================================================
    # SARIMA
    # ====================================================
    if st.button("Run SARIMA Forecast"):
        model = SARIMAX(
            train,
            order=(1,1,1),
            seasonal_order=(1,1,1,7)
        ).fit()

        forecast = model.get_forecast(90).predicted_mean
        st.write(f"RMSE: {np.sqrt(mean_squared_error(test, forecast)):.2f}")

    # ====================================================
    # EXPONENTIAL SMOOTHING
    # ====================================================
    if st.button("Run Exponential Smoothing"):
        model = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add",
            seasonal_periods=30
        ).fit()

        forecast = model.forecast(90)
        st.write(f"RMSE: {np.sqrt(mean_squared_error(test, forecast)):.2f}")

    # ====================================================
    # GARCH VOLATILITY
    # ====================================================
    if st.button("Run GARCH Volatility Forecast"):
        returns = 100 * series.pct_change().dropna()
        model = arch_model(returns, vol="Garch", p=1, q=1)
        res = model.fit(disp="off")

        forecast = res.forecast(horizon=90)
        vol = forecast.variance.values[-1]

        fig = px.line(y=vol, title="Forecasted Volatility")
        st.plotly_chart(fig, use_container_width=True)

    # ====================================================
    # LSTM
    # ====================================================
    if st.button("Run LSTM Forecast"):
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(series.values.reshape(-1,1))

        X, y = [], []
        look_back = 30
        for i in range(len(scaled) - look_back):
            X.append(scaled[i:i + look_back])
            y.append(scaled[i + look_back])

        X, y = np.array(X), np.array(y)

        model = Sequential([
            LSTM(50, input_shape=(look_back, 1)),
            Dense(1)
        ])
        model.compile(loss="mse", optimizer="adam")
        model.fit(X[:-90], y[:-90], epochs=10, batch_size=16, verbose=0)

        preds = model.predict(X[-90:])
        preds = scaler.inverse_transform(preds)

        st.success("LSTM Forecast Completed")
