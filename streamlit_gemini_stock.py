import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
import tempfile
import os
import json
from datetime import datetime, timedelta
import pytz
import uuid

genai.configure(api_key="AIzaSyCbLAGcwBnJYwXQaTQndapcpe4l8OyDjlA")

MODEL_NAME = 'gemini-2.0-flash'
gen_model = genai.GenerativeModel(MODEL_NAME)

st.set_page_config(layout="wide")
st.title("AI-Powered Technical Stock Analysis Dashboard")
st.sidebar.header("Configuration")

tickers_input = st.sidebar.text_input("Enter Stock Tickers (comma-separated):", "RELIANCE.NS")
tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]

end_date_default = datetime.today()
start_date_default = end_date_default - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date", value=start_date_default)
end_date = st.sidebar.date_input("End Date", value=end_date_default)

timeframe_options = ["1m", "5m", "15m", "1h", "1d", "1wk", "1mo"]
selected_timeframes = st.sidebar.multiselect(
    "Select Timeframes:", 
    timeframe_options, 
    default=["1d"]
)

st.sidebar.subheader("Technical Indicators")
indicators = st.sidebar.multiselect(
    "Select Indicators:",
    ["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands", "VWAP", "Volume", "RSI", "MACD"],
    default=["20-Day SMA", "Volume", "RSI", "MACD"]
)

if 'previous_analyses' not in st.session_state:
    st.session_state.previous_analyses = {}
    
if 'chart_metadata' not in st.session_state:
    st.session_state.chart_metadata = {}
    
if 'current_ranges' not in st.session_state:
    st.session_state.current_ranges = {}

def fetch_data(ticker, timeframes):
    """
    Fetch data for multiple timeframes
    
    :param ticker: Stock ticker symbol
    :param timeframes: List of timeframes to fetch
    :return: Dictionary of dataframes for each timeframe
    """
    timeframe_data = {}
    
    for timeframe in timeframes:
        try:
            # Define max days based on timeframe
            max_days = {
                "1m": 7, "5m": 60, "15m": 60, 
                "1h": 730, "1d": 3650, 
                "1wk": 3650, "1mo": 3650
            }
            
            # Adjust start date based on timeframe
            adjusted_start_date = max(start_date, end_date - timedelta(days=max_days.get(timeframe, 365)))
            
            # Fetch data
            data = yf.download(
                ticker, 
                start=adjusted_start_date, 
                end=end_date, 
                interval=timeframe, 
                multi_level_index=False
            )
            
            # Convert UTC to IST
            if data.index.tz is None:
                data.index = data.index.tz_localize('UTC')
            ist = pytz.timezone('Asia/Kolkata')
            data.index = data.index.tz_convert(ist)
            
            if not data.empty:
                timeframe_data[timeframe] = data
            else:
                st.warning(f"No data found for {ticker} with timeframe {timeframe}")
        
        except Exception as e:
            st.error(f"Error fetching data for {ticker} in {timeframe} timeframe: {e}")
    
    return timeframe_data

def calculate_rsi(data, periods=14):
    """
    Calculate RSI manually
    
    :param data: Pandas Series of closing prices
    :param periods: Number of periods to use for RSI calculation
    :return: Pandas Series of RSI values
    """
    # Calculate price changes
    delta = data.diff()
    
    # Separate gains and losses
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    
    # Calculate relative strength
    rs = gain / loss
    
    # Calculate RSI
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate MACD manually
    
    :param data: Pandas Series of closing prices
    :param fast_period: Fast moving average period
    :param slow_period: Slow moving average period
    :param signal_period: Signal line period
    :return: Tuple of MACD line, Signal line, and Histogram
    """
    # Calculate exponential moving averages
    fast_ema = data.ewm(span=fast_period, adjust=False).mean()
    slow_ema = data.ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = fast_ema - slow_ema
    
    # Calculate signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def create_chart(ticker, data, timeframe):
    
    num_rows = 1 + sum([
        "Volume" in indicators, 
        "RSI" in indicators, 
        "MACD" in indicators
    ])

    
    fig = make_subplots(
        rows=num_rows, 
        cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.02,
        subplot_titles=[
            f"{ticker} Candlestick Chart",
            *[ind for ind in ["Volume", "RSI", "MACD"] if ind in indicators]
        ],
        row_heights=[0.6] + [0.4/(num_rows-1)] * (num_rows-1)
    )

    
    last_price = data['Close'].iloc[-1]
    last_price_formatted = f"{last_price:.2f}"

    
    fig.add_trace(
        go.Candlestick(
            x=data.index, 
            open=data['Open'], 
            high=data['High'], 
            low=data['Low'], 
            close=data['Close'], 
            name="Candlestick"
        ),
        row=1, col=1
    )

    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=[last_price] * len(data),
            mode='lines',
            name='Last Traded Price',
            line=dict(color='yellow', width=2, dash='dot'),
            showlegend=True
        ),
        row=1, col=1
    )

    
    indicator_colors = {
        "20-Day SMA": "orange",
        "20-Day EMA": "green",
        "VWAP": "purple"
    }

    for ind in indicators:
        if ind == "20-Day SMA":
            fig.add_trace(
                go.Scatter(
                    x=data.index, 
                    y=data['Close'].rolling(window=20).mean(), 
                    mode='lines', 
                    name='SMA (20)',
                    line=dict(color=indicator_colors[ind], width=2)
                ),
                row=1, col=1
            )
        elif ind == "20-Day EMA":
            fig.add_trace(
                go.Scatter(
                    x=data.index, 
                    y=data['Close'].ewm(span=20).mean(), 
                    mode='lines', 
                    name='EMA (20)',
                    line=dict(color=indicator_colors[ind], width=2)
                ),
                row=1, col=1
            )
        elif ind == "VWAP":
            data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
            fig.add_trace(
                go.Scatter(
                    x=data.index, 
                    y=data['VWAP'], 
                    mode='lines', 
                    name='VWAP',
                    line=dict(color=indicator_colors[ind], width=2)
                ),
                row=1, col=1
            )
        elif ind == "20-Day Bollinger Bands":
            sma = data['Close'].rolling(window=20).mean()
            std = data['Close'].rolling(window=20).std()
            fig.add_trace(
                go.Scatter(
                    x=data.index, 
                    y=sma + 2 * std, 
                    mode='lines', 
                    name='BB Upper',
                    line=dict(color='gray', dash='dot')
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=data.index, 
                    y=sma - 2 * std, 
                    mode='lines', 
                    name='BB Lower',
                    line=dict(color='gray', dash='dot')
                ),
                row=1, col=1
            )

    
    row_counter = 1
    if "Volume" in indicators:
        row_counter += 1
        fig.add_trace(
            go.Bar(
                x=data.index, 
                y=data['Volume'], 
                name='Volume', 
                marker_color='lightgray',
                opacity=0.5
            ),
            row=row_counter, col=1
        )

    
    if "RSI" in indicators:
        row_counter += 1
        rsi = calculate_rsi(data['Close'])
        
        
        fig.add_trace(
            go.Scatter(
                x=rsi.index, 
                y=rsi, 
                mode='lines', 
                name='RSI', 
                line=dict(color='purple', width=2)
            ),
            row=row_counter, col=1
        )
        
        
        fig.add_trace(
            go.Scatter(
                x=rsi.index, 
                y=[70] * len(rsi), 
                mode='lines', 
                name='RSI Overbought',
                line=dict(color='red', dash='dot')
            ),
            row=row_counter, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=rsi.index, 
                y=[30] * len(rsi), 
                mode='lines', 
                name='RSI Oversold',
                line=dict(color='green', dash='dot')
            ),
            row=row_counter, col=1
        )

    
    if "MACD" in indicators:
        row_counter += 1
        macd_line, signal_line, histogram = calculate_macd(data['Close'])
        
        
        fig.add_trace(
            go.Scatter(
                x=macd_line.index, 
                y=macd_line, 
                mode='lines', 
                name='MACD', 
                line=dict(color='blue', width=2)
            ),
            row=row_counter, col=1
        )
        
        
        fig.add_trace(
            go.Scatter(
                x=signal_line.index, 
                y=signal_line, 
                mode='lines', 
                name='Signal', 
                line=dict(color='red', width=1)
            ),
            row=row_counter, col=1
        )
        
        
        fig.add_trace(
            go.Bar(
                x=histogram.index, 
                y=histogram, 
                name='MACD Histogram',
                marker_color=np.where(histogram >= 0, 'green', 'red'),
                opacity=0.5
            ),
            row=row_counter, col=1
        )

    
    fig.update_layout(
        height=1000,
        width=1200,
        title=f"{ticker} Technical Analysis",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        annotations=[
            
            dict(
                xref="paper", 
                yref="paper",
                x=0.02, 
                y=0.98, 
                text=f"Timeframe: {timeframe}", 
                showarrow=False, 
                font=dict(size=14, color="white"),
                align="left", 
                bgcolor="black"
            ),
            
            dict(
                xref="paper",
                yref="y",
                x=1.02,
                y=last_price,
                text=f"Last: ₹{last_price_formatted}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="yellow",
                font=dict(color="yellow", size=12),
                align="left",
                bordercolor="yellow",
                borderwidth=1,
                borderpad=4,
                bgcolor="black"
            )
        ]
    )

    
    for i in range(1, row_counter + 1):
        axis_titles = ["Price", "Volume", "RSI", "MACD"]
        fig.update_yaxes(title_text=axis_titles[i-1], row=i, col=1)

    return fig

def save_visible_chart(ticker, fig, data, visible_range=None):
    """Create a new figure with only the visible range and save it as image"""
    visible_fig = go.Figure(fig)
    
    
    if visible_range and 'xaxis.range[0]' in visible_range and 'xaxis.range[1]' in visible_range:
        x_range = [visible_range['xaxis.range[0]'], visible_range['xaxis.range[1]']]
        visible_fig.update_layout(xaxis_range=x_range)
    
    if visible_range and 'yaxis.range[0]' in visible_range and 'yaxis.range[1]' in visible_range:
        y_range = [visible_range['yaxis.range[0]'], visible_range['yaxis.range[1]']]
        visible_fig.update_layout(yaxis_range=y_range)
    
    
    visible_fig.update_layout(
        updatemenus=[],
        annotations=[]
    )
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        visible_fig.write_image(tmpfile.name, width=1200, height=800)
        tmpfile_path = tmpfile.name
    
    with open(tmpfile_path, "rb") as f:
        image_bytes = f.read()
    os.remove(tmpfile_path)
    
    return image_bytes

def analyze_chart(ticker, fig, data,timeframe, visible_range=None):
    """Generate AI analysis based on the current chart view"""
    image_bytes = save_visible_chart(ticker, fig, data, visible_range)
    image_part = {"data": image_bytes, "mime_type": "image/png"}
    
    
    previous_analyses = ""
    if hasattr(st.session_state, 'ticker_analyses') and ticker in st.session_state.ticker_analyses:
        for timeframe, analysis in st.session_state.ticker_analyses[ticker].items():
            previous_analyses += (
                f"Timeframe {timeframe} Analysis - "
                f"Action: {analysis['action']}, "
                f"Justification: {analysis['justification']}\n"
            )
    
    
   
    analysis_prompt = (
        f"You are a Stock Trader specializing in Technical Analysis. "
        f"The timeframe on the chart is {timeframe} (1m means 1 minute, 1h means 1 hour and 1d means 1 day). "
        f"Previous analyses for this stock across different timeframes:\n{previous_analyses}\n\n"
        f"Analyze the stock chart for {ticker} based on its candlestick chart and technical indicators. "
        f"If the timeframe is 1m state your intraday suggestion specifying it's an intraday suggestion. "
        f"If timeframe on the chart is not 1m or 5m specify the holding period for the recommendation. "
        f"What is the price at which I should buy/sell the stock and what will be the exit level? "
        f"Also specify what additional info you might need in the chart for a better analysis. "
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
    
    
    return result

if st.sidebar.button("Fetch Data"):
    # Fetch data for all tickers and timeframes
    stock_data = {}
    for ticker in tickers:
        ticker_timeframe_data = fetch_data(ticker, selected_timeframes)
        if ticker_timeframe_data:
            stock_data[ticker] = ticker_timeframe_data
    
    # Store in session state
    st.session_state["stock_data"] = stock_data
    
    # Reset previous analyses and metadata
    st.session_state.ticker_analyses = {}
    st.session_state.previous_analyses = {}
    st.session_state.chart_metadata = {}
    st.session_state.current_ranges = {}
    st.session_state.data_fetched_timestamp = datetime.now()
    st.success("Stock data loaded successfully.")

if 'ticker_analyses' not in st.session_state:
    st.session_state.ticker_analyses = {}

if "stock_data" in st.session_state and st.session_state["stock_data"]:
    current_timeframe = st.sidebar.selectbox(
        "Select Timeframe to Display", 
        selected_timeframes
    )
    tabs = st.tabs(["Overall Summary"] + list(st.session_state["stock_data"].keys()))
    
    overall_summary = []
    
    for i, ticker in enumerate(st.session_state["stock_data"]):
        data = st.session_state["stock_data"][ticker][current_timeframe]
        fig = create_chart(ticker, data,current_timeframe)
        
        
        chart_key = f"chart_{ticker}"
        
        with tabs[i + 1]:
            st.subheader(f"Analysis for {ticker}")

            available_timeframes = list(st.session_state["stock_data"][ticker].keys())
            
            if ticker not in st.session_state.ticker_analyses:
                st.session_state.ticker_analyses[ticker] = {}
            
            # Display previous analyses for this ticker
            if st.session_state.ticker_analyses.get(ticker):
                st.subheader("Previous Analyses")
                for tf, prev_analysis in st.session_state.ticker_analyses[ticker].items():
                    if tf != current_timeframe:
                        st.write(f"**{tf} Timeframe Analysis:**")
                        st.info(f"Action: {prev_analysis['action']}")
                        st.write(f"Justification: {prev_analysis['justification']}")
                        st.write(f"Analyzed on: {prev_analysis['timestamp']}")
            
            chart_container = st.container()
            
            
    
            
            
            with chart_container:
                
                col1, col2 = st.columns(2)
                with col1:
                    range_x_min = st.text_input(f"X Min for {ticker}", value="", key=f"x_min_{ticker}", label_visibility="collapsed")
                    range_y_min = st.text_input(f"Y Min for {ticker}", value="", key=f"y_min_{ticker}", label_visibility="collapsed")
                with col2:
                    range_x_max = st.text_input(f"X Max for {ticker}", value="", key=f"x_max_{ticker}", label_visibility="collapsed")
                    range_y_max = st.text_input(f"Y Max for {ticker}", value="", key=f"y_max_{ticker}", label_visibility="collapsed")
                
                
                components_js = f"""
                <script>
                document.addEventListener('DOMContentLoaded', function() {{
                    const observer = new MutationObserver(function(mutations) {{
                        mutations.forEach(function(mutation) {{
                            if (mutation.addedNodes.length) {{
                                const plotDiv = document.querySelector('div[id^="chart-{chart_key}"]');
                                if (plotDiv) {{
                                    plotDiv.on('plotly_relayout', function(eventData) {{
                                        if (eventData['xaxis.range[0]'] && eventData['xaxis.range[1]']) {{
                                            document.querySelector('input[data-testid="stTextInput"][aria-label="X Min for {ticker}"]').value = eventData['xaxis.range[0]'];
                                            document.querySelector('input[data-testid="stTextInput"][aria-label="X Max for {ticker}"]').value = eventData['xaxis.range[1]'];
                                        }}
                                        if (eventData['yaxis.range[0]'] && eventData['yaxis.range[1]']) {{
                                            document.querySelector('input[data-testid="stTextInput"][aria-label="Y Min for {ticker}"]').value = eventData['yaxis.range[0]'];
                                            document.querySelector('input[data-testid="stTextInput"][aria-label="Y Max for {ticker}"]').value = eventData['yaxis.range[1]'];
                                        }}
                                        const button = document.querySelector('button[kind="secondary"][data-testid="baseButton-secondary"]');
                                        if (button) {{
                                            button.click();
                                        }}
                                    }});
                                    observer.disconnect();
                                }}
                            }}
                        }});
                    }});
                    
                    observer.observe(document.body, {{ childList: true, subtree: true }});
                }});
                </script>
                """
                
                st.components.v1.html(components_js, height=0)
                chart = st.plotly_chart(fig, use_container_width=True, key=chart_key)
            
            
            visible_range = {}
            if range_x_min and range_x_max:
                visible_range['xaxis.range[0]'] = range_x_min
                visible_range['xaxis.range[1]'] = range_x_max
            if range_y_min and range_y_max:
                visible_range['yaxis.range[0]'] = float(range_y_min)
                visible_range['yaxis.range[1]'] = float(range_y_max)
            
            st.session_state.current_ranges[ticker] = visible_range

            
            
            
            if st.button("Analyze Chart", key=f"analyze_{ticker}"):
                with st.spinner(f"Analyzing {ticker}..."):
                    analysis = analyze_chart(ticker, fig, data,current_timeframe, st.session_state.current_ranges[ticker])
                    
                    st.session_state.ticker_analyses[ticker][current_timeframe] = {
                        'action': analysis['action'],
                        'justification': analysis['justification'],
                        'timestamp': datetime.now()
                    }
                    st.subheader("AI Analysis")
                    st.info(f"**Action:** {analysis['action']}")
                    st.write(f"**Justification:** {analysis['justification']}")
                    
                    
                    overall_summary.append({
                        "ticker": ticker, 
                        "action": analysis['action'],
                        "justification": analysis['justification']
                    })
    
    
    with tabs[0]:
        st.subheader("Overall Market Summary")
        if overall_summary:
            for item in overall_summary:
                st.write(f"**{item['ticker']}**: {item['action']}")
            
            
            if st.button("Generate Market Overview"):
                with st.spinner("Generating market overview..."):
                    summary_prompt = (
                        f"You are a Stock Market Analyst. Based on the following individual stock analyses, "
                        f"provide an overall market sentiment and summary. Consider any patterns or contradictions "
                        f"in the analyses. Here are the individual analyses:\n\n"
                    )
                    
                    for item in overall_summary:
                        summary_prompt += f"Stock: {item['ticker']}, Action: {item['action']}, Justification: {item['justification']}\n\n"
                    
                    summary_prompt += "Provide a concise market overview based on these analyses."
                    
                    response = gen_model.generate_content(summary_prompt)
                    st.write("### Market Overview")
                    st.write(response.text)
        else:
            st.info("Analyze individual stocks to generate an overall summary")
else:
    st.info("Please enter stock ticker(s) and fetch data to begin analysis")
