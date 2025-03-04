import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
import tempfile
import os
import json
from datetime import datetime, timedelta
import pytz

# Configure API Key
genai.configure(api_key="AIzaSyCbLAGcwBnJYwXQaTQndapcpe4l8OyDjlA")

MODEL_NAME = 'gemini-2.0-flash'
gen_model = genai.GenerativeModel(MODEL_NAME)

st.set_page_config(layout="wide")
st.title("AI-Powered Technical Stock Analysis Dashboard")
st.sidebar.header("Configuration")

# User Inputs
tickers_input = st.sidebar.text_input("Enter Stock Tickers (comma-separated):", "RELIANCE.NS,HDFCBANK.NS,GOOGL")
tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]

end_date_default = datetime.today()
start_date_default = end_date_default - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date", value=start_date_default)
end_date = st.sidebar.date_input("End Date", value=end_date_default)
#end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
# Timeframe selection
timeframe_options = ["1m", "5m", "15m", "1h", "1d", "1wk", "1mo"]
timeframe = st.sidebar.selectbox("Select Timeframe:", timeframe_options, index=4)

# Technical Indicators selection
st.sidebar.subheader("Technical Indicators")
indicators = st.sidebar.multiselect(
    "Select Indicators:",
    ["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands", "VWAP"],
    default=["20-Day SMA"]
)

def fetch_data(ticker):
    try:
        max_days = {"1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60, "1h": 730, "1d": 3650, "1wk": 3650, "1mo": 3650}
        adjusted_start_date = max(start_date, end_date - timedelta(days=max_days.get(timeframe, 365)))
        print(type(adjusted_start_date))
        #adjusted_start_date = datetime.strptime(adjusted_start_date, "%Y-%m-%d").date()
        data = yf.download(ticker, start=adjusted_start_date, end=end_date, interval=timeframe,multi_level_index=False)

# Convert UTC to IST
        if data.index.tz is None:  # If it's timezone-naive
            data.index = data.index.tz_localize('UTC')
        ist = pytz.timezone('Asia/Kolkata')
        data.index = data.index.tz_convert(ist)
        if data.empty:
            st.warning(f"No data found for {ticker} with timeframe {timeframe}. Try a different timeframe or date range.")
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

def identify_support_resistance(data):
    levels = []
    for i in range(2, len(data) - 2):
        if data['Low'][i] < data['Low'][i-1] and data['Low'][i] < data['Low'][i+1]:
            levels.append((data.index[i], data['Low'][i]))
        if data['High'][i] > data['High'][i-1] and data['High'][i] > data['High'][i+1]:
            levels.append((data.index[i], data['High'][i]))
    return levels

def analyze_ticker(ticker, data):
    fig = go.Figure(data=[
        go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Candlestick")
    ])
    
    for ind in indicators:
        if ind == "20-Day SMA":
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'].rolling(window=20).mean(), mode='lines', name='SMA (20)'))
        elif ind == "20-Day EMA":
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'].ewm(span=20).mean(), mode='lines', name='EMA (20)'))
        elif ind == "20-Day Bollinger Bands":
            sma = data['Close'].rolling(window=20).mean()
            std = data['Close'].rolling(window=20).std()
            fig.add_trace(go.Scatter(x=data.index, y=sma + 2 * std, mode='lines', name='BB Upper'))
            fig.add_trace(go.Scatter(x=data.index, y=sma - 2 * std, mode='lines', name='BB Lower'))
        elif ind == "VWAP":
            data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
            fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP'))
    
    # levels = identify_support_resistance(data)
    # for level in levels:
    #     fig.add_hline(y=level[1], line=dict(color="red", width=1, dash="dot"))
    
    # fig.update_layout(xaxis_rangeslider_visible=False)
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        fig.write_image(tmpfile.name)
        tmpfile_path = tmpfile.name
    with open(tmpfile_path, "rb") as f:
        image_bytes = f.read()
    os.remove(tmpfile_path)

    image_part = {"data": image_bytes, "mime_type": "image/png"}

    analysis_prompt = (
        f"You are a Stock Trader specializing in Technical Analysis. "
        f"Analyze the stock chart for {ticker} based on its candlestick chart and technical indicators. "
        f"If the timeframe on the chart is 1m or 5m give an intraday view or else specify the holding period for the recommendation"
        f"What is the price at which i should buy/sell the stock and what will be the exit level? "
        f"Also specify what additional info you might need in the chart for a better analysis."
        f"Provide a detailed analysis and recommendation. Return JSON with 'action' and 'justification'."

        
    )

    contents = [
        {"role": "user", "parts": [analysis_prompt]},
        {"role": "user", "parts": [image_part]}
    ]

    response = gen_model.generate_content(contents=contents)
    
    try:
        result_text = response.text
        json_start_index = result_text.find('{')
        json_end_index = result_text.rfind('}') + 1
        json_string = result_text[json_start_index:json_end_index]
        result = json.loads(json_string)
    except Exception as e:
        result = {"action": "Error", "justification": f"Error parsing response: {e}."}
    
    return fig, result

if st.sidebar.button("Fetch Data"):
    stock_data = {ticker: fetch_data(ticker) for ticker in tickers}
    stock_data = {k: v for k, v in stock_data.items() if v is not None}
    st.session_state["stock_data"] = stock_data
    st.success("Stock data loaded successfully.")

if "stock_data" in st.session_state and st.session_state["stock_data"]:
    tabs = st.tabs(["Overall Summary"] + list(st.session_state["stock_data"].keys()))
    
    for i, ticker in enumerate(st.session_state["stock_data"]):
        data = st.session_state["stock_data"][ticker]
        fig, result = analyze_ticker(ticker, data)
        with tabs[i + 1]:
            st.subheader(f"Analysis for {ticker}")
            st.plotly_chart(fig)
            st.write("**Recommendation:**")
            st.write(result.get("action", "No justification provided."))
            st.write("**Justification:**")
            st.write(result.get("justification", "No justification provided."))
