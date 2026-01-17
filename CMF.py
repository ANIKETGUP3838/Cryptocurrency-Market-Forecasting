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
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("📈 Cryptocurrency Market Forecasting App")

tab1, tab2 = st.tabs(["📄 Project Summary", "📊 Forecasting App"])

with tab1:
    st.markdown("""
    Cryptocurrency forecasting using:
    ARIMA, SARIMA, Exponential Smoothing, GARCH, LSTM
    """)

with tab2:

    uploaded_file = st.sidebar.file_uploader(
        "Upload crypto-markets.csv",
        type=["csv"],
        key="file_uploader"
    )

    # 🔴 RESET STATE WHEN NEW FILE IS UPLOADED
    if uploaded_file is not None and "last_file" not in st.session_state:
        st.session_state.clear()
        st.session_state["last_file"] = uploaded_file.name

    if uploaded_file is None:
        st.stop()

    # ---------- LOAD DATA ----------
    crypto_df = pd.read_csv(uploaded_file)

    # 🔴 CLEAN COLUMN NAMES
    crypto_df.columns = (
        crypto_df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    columns = list(crypto_df.columns)

    st.sidebar.markdown("### Dataset Columns")
    st.sidebar.write(columns)

    # ---------- DATE COLUMN ----------
    date_col = st.sidebar.selectbox(
        "Select Date Column",
        options=columns,
        key="date_col_select"
    )

    if date_col not in crypto_df.columns:
        st.error("Invalid date column selected")
        st.stop()

    crypto_df[date_col] = pd.to_datetime(
        crypto_df[date_col],
        errors="coerce"
    )
    crypto_df = crypto_df.dropna(subset=[date_col])
    crypto_df = crypto_df.sort_values(by=date_col)

    # ---------- CRYPTO IDENTIFIER COLUMN ----------
    crypto_col = st.sidebar.selectbox(
        "Select Crypto Identifier Column",
        options=columns,
        key="crypto_col_select"
    )

    if crypto_col not in crypto_df.columns:
        st.error("Invalid crypto column selected")
        st.stop()

    crypto_value = st.sidebar.selectbox(
        "Select Cryptocurrency",
        options=sorted(crypto_df[crypto_col].astype(str).unique()),
        key="crypto_value_select"
    )

    filtered_df = crypto_df[
        crypto_df[crypto_col].astype(str) == crypto_value
    ].copy()

    filtered_df.set_index(date_col, inplace=True)

    # ---------- TARGET COLUMN ----------
    numeric_cols = filtered_df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    if not numeric_cols:
        st.error("No numeric columns available for forecasting")
        st.stop()

    target_col = st.sidebar.selectbox(
        "Select Target Variable",
        options=numeric_cols,
        key="target_col_select"
    )

    if target_col not in filtered_df.columns:
        st.error("Invalid target column selected")
        st.stop()

    series = filtered_df[target_col].dropna()

    if len(series) < 120:
        st.warning("At least 120 data points are required")
        st.stop()

    # ---------- PLOT ----------
    st.plotly_chart(px.line(series, title="Time Series"), use_container_width=True)

    # ---------- DECOMPOSITION ----------
    decomposition = sm.tsa.seasonal_decompose(series, model="additive", period=30)

    fig = make_subplots(rows=4, cols=1)
    fig.add_trace(go.Scatter(y=decomposition.observed), 1, 1)
    fig.add_trace(go.Scatter(y=decomposition.trend), 2, 1)
    fig.add_trace(go.Scatter(y=decomposition.seasonal), 3, 1)
    fig.add_trace(go.Scatter(y=decomposition.resid), 4, 1)
    fig.update_layout(height=800)
    st.plotly_chart(fig, use_container_width=True)

    # ---------- ADF ----------
    st.write("ADF p-value:", adfuller(series)[1])

    train = series[:-90]
    test = series[-90:]

    if st.button("Run ARIMA", key="arima_btn"):
        model = ARIMA(train, order=(5, 1, 0)).fit()
        forecast = model.forecast(90)
        st.write("RMSE:", np.sqrt(mean_squared_error(test, forecast)))

    if st.button("Run SARIMA", key="sarima_btn"):
        model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,7)).fit()
        forecast = model.get_forecast(90).predicted_mean
        st.write("RMSE:", np.sqrt(mean_squared_error(test, forecast)))

    if st.button("Run Exponential Smoothing", key="exp_btn"):
        model = ExponentialSmoothing(
            train, trend="add", seasonal="add", seasonal_periods=30
        ).fit()
        forecast = model.forecast(90)
        st.write("RMSE:", np.sqrt(mean_squared_error(test, forecast)))

    if st.button("Run GARCH", key="garch_btn"):
        returns = 100 * series.pct_change().dropna()
        model = arch_model(returns, vol="Garch", p=1, q=1)
        res = model.fit(disp="off")
        vol = res.forecast(horizon=90).variance.values[-1]
        st.plotly_chart(px.line(y=vol), use_container_width=True)

    if st.button("Run LSTM", key="lstm_btn"):
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(series.values.reshape(-1, 1))

        X, y = [], []
        for i in range(len(scaled) - 30):
            X.append(scaled[i:i+30])
            y.append(scaled[i+30])

        X, y = np.array(X), np.array(y)

        model = Sequential([
            LSTM(50, input_shape=(30, 1)),
            Dense(1)
        ])
        model.compile(loss="mse", optimizer="adam")
        model.fit(X[:-90], y[:-90], epochs=10, batch_size=16, verbose=0)

        preds = scaler.inverse_transform(model.predict(X[-90:]))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test.index, y=test, name="Actual"))
        fig.add_trace(go.Scatter(x=test.index, y=preds.flatten(), name="Forecast"))
        st.plotly_chart(fig, use_container_width=True)
