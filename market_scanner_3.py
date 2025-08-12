#!/usr.bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator
from ta.volume import VolumeWeightedAveragePrice
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import datetime as dt
import json
import requests
import time
import pytz

# NOTE: The API key for the Gemini API is intentionally left as an empty string.
# The Canvas environment will automatically handle the API key for the fetch call.
API_KEY = "AIzaSyChisj-FzBOwejY0aVd8cSlF8mPSTBaC9Y"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"

# --- Streamlit App Configuration with Layout Fixes ---
st.set_page_config(
    layout="wide",
    page_title="Stock Scanner",
    initial_sidebar_state="expanded"
)

# --- CSS Fixes for Layout Issues ---
st.markdown("""
<style>
    /* Main app container fixes */
    .stApp {
        margin-top: -2rem;
        padding-top: 0;
    }

    /* Title section spacing */
    .dashboard-title-container {
        margin-bottom: 2rem;
    }

    .dashboard-title {
        font-size: 3rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.5rem !important;
    }

    .dashboard-subtitle {
        font-size: 1.25rem;
        font-weight: 400;
        color: #cccccc;
        margin-top: 0 !important;
        margin-bottom: 2rem !important;
    }

    /* Fix for overlapping elements */
    .st-emotion-cache-1v0mbdj,
    .st-emotion-cache-1q7spjk {
        margin-top: 0 !important;
        margin-bottom: 1rem !important;
    }

    /* Section header spacing */
    .section-header {
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }

    /* Card styling for individual scanners */
    .st-emotion-cache-1wivd2v {
        background-color: #2a2a2a;
        border: 1px solid #444444;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        margin-top: 1rem !important;
        height: 280px; /* Fixed height for uniformity */
        display: flex;
        flex-direction: column;
        overflow-y: auto; /* Adds a scrollbar if content overflows */
    }

    /* Score colors */
    .positive-score {
        color: #38b2ac;
        font-weight: 700;
    }
    .negative-score {
        color: #e53e3e;
        font-weight: 700;
    }

    /* Table styling */
    .dataframe {
        margin-top: 1rem !important;
    }

    div[data-testid="stExpander"] button {
        font-size: 3rem;
        font-weight: 700;
    }

    .st-emotion-cache-16p6y9p {
        border-radius: 12px;
    }

    .st-emotion-cache-1wivd2v > div:first-child {
        padding-bottom: 0 !important;
    }

    .st-emotion-cache-1wivd2v > div:last-child {
        padding-top: 0 !important;
    }
    
    /* New Sentiment styling */
    .sentiment-positive {
        background-color: #03a84e; /* A nice green */
        color: white;
        padding: 4px 8px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
    }
    .sentiment-negative {
        background-color: #d11239; /* A nice red */
        color: white;
        padding: 4px 8px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
    }
    .sentiment-neutral {
        background-color: #555555; /* A dark gray */
        color: white;
        padding: 4px 8px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
    }
    .sentiment-label {
        font-size: 1.2rem;
        font-weight: 600;
    }

    /* Huge card styling for custom analysis section */
    .huge-card {
        background-color: #2a2a2a;
        border: 1px solid #444444;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        margin-top: 2rem !important;
    }
</style>
""", unsafe_allow_html=True)

# --- App Header ---
st.markdown("""
<div class="dashboard-title-container">
    <h1 class="dashboard-title">Enhanced Stock Scanner</h1>
    <p class="dashboard-subtitle">Interactive dashboard to analyze a custom selection of stocks using technical indicators.</p>
</div>
""", unsafe_allow_html=True)

# --- Dynamic Tickers and Company Names Fetching using official NSE CSVs ---
@st.cache_data(ttl=3600 * 6)
def get_nifty_indices_data():
    """
    Fetches real-time Nifty 50 and Nifty 100 stock data from official NSE CSVs.
    """
    nifty50_url = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
    nifty100_url = "https://archives.nseindia.com/content/indices/ind_nifty100list.csv"

    ticker_to_name = {}
    nifty50_tickers = []
    nifty100_tickers = []

    try:
        nifty50_df = pd.read_csv(nifty50_url)
        nifty50_tickers = [symbol + '.NS' for symbol in nifty50_df['Symbol'].tolist()]
        for _, row in nifty50_df.iterrows():
            ticker_to_name[row['Symbol'] + '.NS'] = row['Company Name']

        nifty100_df = pd.read_csv(nifty100_url)
        nifty100_tickers = [symbol + '.NS' for symbol in nifty100_df['Symbol'].tolist()]
        for _, row in nifty100_df.iterrows():
            ticker_to_name[row['Symbol'] + '.NS'] = row['Company Name']

    except Exception as e:
        st.error(f"Error fetching data from NSE archives: {e}")
        return [], [], {}

    master_tickers = sorted(list(set(nifty50_tickers + nifty100_tickers)))
    return nifty50_tickers, nifty100_tickers, ticker_to_name

# Fetch the data and populate global variables
with st.spinner("Fetching the latest Nifty 50 and Nifty 100 stock lists..."):
    nifty50_tickers, nifty100_tickers, ticker_to_name = get_nifty_indices_data()

master_tickers = sorted(list(set(nifty50_tickers + nifty100_tickers)))
master_names = sorted([ticker_to_name.get(t, t.split('.')[0]) for t in master_tickers])
name_to_ticker = {v: k for k, v in ticker_to_name.items()}

# --- Data Download Function (NOW WITH CACHING) ---
@st.cache_data(ttl=300) # Cache for 5 minutes
def get_stock_data(tickers, start_date, end_date):
    try:
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            group_by='ticker',
            auto_adjust=True,
            threads=True,
            progress=False
        )
        return data
    except Exception as e:
        st.error(f"Failed to download data from Yahoo Finance. Error: {e}")
        return pd.DataFrame()

# --- ORIGINAL SCORING FUNCTIONS ---
def ma50_score(df):
    if len(df) < 200:
        return 0.0
    ma50_series = SMAIndicator(close=df['Close'], window=50).sma_indicator()
    ma200_series = SMAIndicator(close=df['Close'], window=200).sma_indicator()
    if pd.isna(ma50_series.iloc[-1]) or pd.isna(ma200_series.iloc[-1]):
        return 0.0
    ma50 = ma50_series.iloc[-1]
    ma50_5ago = ma50_series.iloc[-5] if len(ma50_series) >= 5 else ma50
    ma200 = ma200_series.iloc[-1]
    close = df['Close'].iloc[-1]
    
    dist_score = 0
    if ma50 != 0:
        dist = (close - ma50) / ma50
        dist_score = max(-1, min(1, dist * 5))
    
    slope_score = 0
    if ma50_5ago != 0:
        slope = (ma50 - ma50_5ago) / ma50_5ago
        slope_score = max(-1, min(1, slope * 100))
    
    regime = 1 if ma50 > ma200 else -1
    score = 0.5 * dist_score + 0.3 * slope_score + 0.2 * regime
    return max(-1, min(1, score))

def rsi_score_momentum(df, rsi_period=14, lookback=20):
    if len(df) < rsi_period + lookback:
        return 0.0
    rsi_series = RSIIndicator(close=df['Close'], window=rsi_period).rsi()
    rsi_change = rsi_series.diff()
    change = rsi_change.iloc[-1]
    stdev = rsi_change.rolling(lookback).std().iloc[-1]
    if pd.isna(stdev) or stdev == 0:
        return 0.0
    score = change / (2 * stdev)
    return max(-1, min(1, score))

def vol_score(df, lookback_vol=20):
    if len(df) < lookback_vol + 1:
        return 0.0
    avg_vol = df['Volume'].rolling(window=lookback_vol).mean().iloc[-1]
    curr_vol = df['Volume'].iloc[-1]
    ratio = curr_vol / avg_vol if avg_vol != 0 else 0
    if df['Close'].iloc[-1] > df['Close'].iloc[-2]:
        trend_sign = 1
    elif df['Close'].iloc[-1] < df['Close'].iloc[-2]:
        trend_sign = -1
    else:
        trend_sign = 0
    capped_ratio = min(ratio, 2.0)
    score_volume = capped_ratio * trend_sign
    return max(-1, min(1, score_volume))

@st.cache_data
def compute_scores(data_frame, selected_tickers):
    results = []
    for ticker in selected_tickers:
        if ticker in data_frame.columns.get_level_values(0):
            df_ticker = data_frame[ticker].dropna()
            if not df_ticker.empty and len(df_ticker) > 200:
                ma50_s = ma50_score(df_ticker)
                rsi_s = rsi_score_momentum(df_ticker)
                vol_s = vol_score(df_ticker)
                result = {
                    'Ticker': ticker,
                    'Company Name': ticker_to_name.get(ticker, ticker),
                    'MA50 Score': ma50_s,
                    'RSI Score': rsi_s,
                    'Volume Score': vol_s,
                }
                result['Final Score'] = (result['MA50 Score'] + result['RSI Score'] + result['Volume Score']) / 3
                results.append(result)
    
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values('Final Score', ascending=False).reset_index(drop=True)

# --- NEW SCANNER FUNCTIONS ---
def find_52_week_high(df_all):
    results = []
    for ticker in df_all.columns.get_level_values(0).unique():
        df = df_all[ticker].dropna()
        if len(df) >= 252:
            high_52_weeks = df['High'].iloc[-252:].max()
            if df['Close'].iloc[-1] >= high_52_weeks:
                results.append(ticker)
    return results

def find_open_equals_high_low(df_all):
    results = []
    for ticker in df_all.columns.get_level_values(0).unique():
        df = df_all[ticker].dropna()
        if not df.empty:
            open_price = df['Open'].iloc[-1]
            high_price = df['High'].iloc[-1]
            low_price = df['Low'].iloc[-1]
            
            if open_price == high_price or open_price == low_price:
                results.append(ticker)
    return results

def find_bullish_engulfing(df_all):
    results = []
    for ticker in df_all.columns.get_level_values(0).unique():
        df = df_all[ticker].dropna()
        if len(df) >= 2:
            prev_close = df['Close'].iloc[-2]
            prev_open = df['Open'].iloc[-2]
            curr_close = df['Close'].iloc[-1]
            curr_open = df['Open'].iloc[-1]

            if (prev_close < prev_open and
                curr_close > curr_open and
                curr_open < prev_close and
                curr_close > prev_open):
                results.append(ticker)
    return results
    
# --- NEW: Volume Breaker Scanner ---
def find_volume_breaker(df_all, multiplier=2, window=20):
    results = []
    for ticker in df_all.columns.get_level_values(0).unique():
        df = df_all[ticker].dropna()
        if len(df) > window:
            current_volume = df['Volume'].iloc[-1]
            average_volume = df['Volume'].iloc[-window-1:-1].mean()
            
            if average_volume > 0 and current_volume > (average_volume * multiplier):
                results.append(ticker)
    return results


# --- ORIGINAL: 200 EMA Crossover ---
def find_200_ema_crossover(df_all):
    results = []
    for ticker in df_all.columns.get_level_values(0).unique():
        df = df_all[ticker].dropna()
        if len(df) >= 200:
            df['EMA200'] = EMAIndicator(close=df['Close'], window=200).ema_indicator()
            if (df['Close'].iloc[-1] > df['EMA200'].iloc[-1] and
                df['Close'].iloc[-2] < df['EMA200'].iloc[-2]):
                results.append(ticker)
    return results

# --- NEW SCANNER: Golden Cross (SMA 50/200) ---
def find_golden_cross(df_all):
    results = []
    for ticker in df_all.columns.get_level_values(0).unique():
        df = df_all[ticker].dropna()
        if len(df) >= 200:
            ma50 = SMAIndicator(close=df['Close'], window=50).sma_indicator()
            ma200 = SMAIndicator(close=df['Close'], window=200).sma_indicator()
            
            # Ensure we have enough data points for the cross
            if len(ma50) > 1 and len(ma200) > 1:
                if ma50.iloc[-1] > ma200.iloc[-1] and ma50.iloc[-2] <= ma200.iloc[-2]:
                    results.append(ticker)
    return results


# --- Main Scanners Dictionary ---
SCANNERS = {
    'custom_score': {
        'title': 'Custom Trend Score',
        'function': 'compute_scores',
        'icon': '📊',
        'description': 'A composite score based on MA50 trend, RSI momentum, and Volume strength.'
    },
    'golden_cross': {
        'title': 'Golden Cross (50/200 SMA)',
        'function': 'find_golden_cross',
        'icon': '✨',
        'description': 'The 50-day SMA has crossed above the 200-day SMA, a classic long-term bullish signal.'
    },
    '52_week_high': {
        'title': '52 Week High',
        'function': 'find_52_week_high',
        'icon': '🚀',
        'description': 'Stocks trading at their 52-week high price, indicating strong bullish momentum.'
    },
    'open_high_low': {
        'title': 'Open = High/Low',
        'function': 'find_open_equals_high_low',
        'icon': '⚡',
        'description': 'Stocks with strong intraday momentum (open price is also high or low).'
    },
    'bullish_engulfing': {
        'title': 'Bullish Engulfing',
        'icon': '🐂',
        'function': 'find_bullish_engulfing',
        'description': 'A classic bullish reversal candlestick pattern.'
    },
    'volume_breaker': {
        'title': 'Volume Breaker',
        'icon': '🔊',
        'function': 'find_volume_breaker',
        'description': 'Stocks with a significant surge in trading volume (more than 2x the 20-day average).'
    },
    '200_ema_crossover': {
        'title': '200 EMA Crossover',
        'function': 'find_200_ema_crossover',
        'icon': '📈',
        'description': 'Stocks with a recent bullish crossover above the 200-day EMA, signaling a long-term trend change.'
    },
}

# --- Function to call the LLM API and get JSON response ---
@st.cache_data(ttl=3600*4)
def get_llm_summary(prompt, retry_count=3, backoff_factor=1.0):
    if not API_KEY:
        return {"error": "API key not provided"}

    headers = {'Content-Type': 'application/json'}
    payload = {
        'contents': [
            {'parts': [{'text': prompt}]}
        ],
        'generationConfig': {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "summary": {"type": "STRING"},
                    "key_points": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"}
                    },
                    "sentiment": {"type": "STRING", "enum": ["Positive", "Neutral", "Negative"]}
                }
            }
        }
    }

    for i in range(retry_count):
        try:
            response = requests.post(f"{GEMINI_API_URL}?key={API_KEY}", headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            result = response.json()
            
            if 'candidates' in result and result['candidates']:
                json_string = result['candidates'][0]['content']['parts'][0]['text']
                json_string = json_string.strip().lstrip('```json').rstrip('```')
                
                return json.loads(json_string)
            else:
                return {"error": "Could not generate a summary. The model response was empty or malformed."}
        
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            if i < retry_count - 1:
                time.sleep(backoff_factor * (2 ** i))
            else:
                return {"error": f"API call failed after multiple retries: {e}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred: {e}"}
    return {"error": "Unknown error."}

# --- Helper function to check if the market is open (IST) ---
def is_market_open():
    ist = pytz.timezone('Asia/Kolkata')
    now = dt.datetime.now(ist)
    start_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
    end_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    is_weekday = now.weekday() < 5 # Monday is 0, Friday is 4
    is_open = now >= start_time and now <= end_time
    
    return is_weekday and is_open

# --- Initialize session state for navigation and data ---
if 'custom_analysis_tickers' not in st.session_state:
    st.session_state['custom_analysis_tickers'] = []
if 'analysis_run' not in st.session_state:
    st.session_state['analysis_run'] = False
if 'custom_scanner_df' not in st.session_state:
    st.session_state['custom_scanner_df'] = pd.DataFrame()
if 'master_data' not in st.session_state:
    st.session_state['master_data'] = pd.DataFrame()
if 'last_run_time' not in st.session_state:
    st.session_state['last_run_time'] = dt.datetime.now()

# =========================================================================
# --- CENTRALIZED DATA FETCHING LOGIC ---
# This block runs first to ensure data is consistent before any component renders.
# =========================================================================
all_tickers_to_fetch = list(set(nifty100_tickers + st.session_state.get('custom_analysis_tickers', [])))
master_data = pd.DataFrame()
force_data_refresh = False

# Logic for live data refresh
if is_market_open():
    if (dt.datetime.now() - st.session_state['last_run_time']).total_seconds() > 300: # 5 minutes
        force_data_refresh = True
        st.info("🔄 Market is open. Refreshing live data...")
    else:
        master_data = st.session_state['master_data']
else:
    # If market is closed, only fetch once at the start
    if st.session_state['master_data'].empty:
        force_data_refresh = True
        st.info("Market is closed. Fetching end-of-day data.")
    else:
        master_data = st.session_state['master_data']

# Fetch data if a refresh is needed or if the user triggered it from the form
if force_data_refresh or 'data_refreshed_by_form' in st.session_state:
    if 'data_refreshed_by_form' in st.session_state:
        del st.session_state['data_refreshed_by_form'] # Clear the flag
    
    start_date = dt.date.today() - dt.timedelta(days=365)
    end_date = dt.date.today()
    master_data = get_stock_data(all_tickers_to_fetch, start_date, end_date)
    st.session_state['master_data'] = master_data
    st.session_state['last_run_time'] = dt.datetime.now()

if master_data.empty:
    st.warning("Could not fetch stock data. Please check your internet connection or the stock tickers.")
    st.stop() # Stop the app if no data is available

# =========================================================================
# --- END OF CENTRALIZED DATA FETCHING LOGIC ---
# =========================================================================

# --- Live Trigger-Based Scanners (Always run on Nifty 100) ---
st.markdown("## 📋 Scanner Dashboard")
with st.container(border=True):
    st.markdown("### Live Triggers on Nifty 100 Stocks")
    
    scanner_keys = list(SCANNERS.keys())
    cols = st.columns(3)
    col_index = 0
    
    # We need to run the custom_score scanner on Nifty 100 data to populate the cards
    nifty100_scores_df = compute_scores(master_data, nifty100_tickers)
    
    for key in scanner_keys:
        scanner_info = SCANNERS[key]
        
        with st.spinner(f"Running {scanner_info['title']} scanner..."):
            if key == 'custom_score':
                # For custom_score, we display the top 3
                top_stocks = nifty100_scores_df.nlargest(3, 'Final Score')['Ticker'].tolist()
            else:
                func = globals()[scanner_info['function']]
                top_stocks = func(master_data)
        
        with cols[col_index]:
            with st.container(border=True):
                st.markdown(f"### {scanner_info['icon']} {scanner_info['title']}")
                st.write(scanner_info['description'])
                st.markdown("---")
                
                if top_stocks:
                    for ticker in top_stocks[:3]:
                        button_clicked = st.button(ticker_to_name.get(ticker, ticker), key=f"details_{key}_{ticker}", use_container_width=True)
                        # --- NEW: Make the buttons interactive ---
                        if button_clicked:
                            st.session_state['custom_analysis_tickers'] = [ticker]
                            st.session_state['analysis_run'] = True
                            st.session_state['data_refreshed_by_form'] = True # Set flag to force data refresh
                            st.rerun()
                else:
                    st.info(f"No stocks found for this scanner.")
        
        col_index = (col_index + 1) % 3


# --- Custom Quantitative Analysis Sections ---
st.markdown("### Custom Quantitative Analysis")
with st.container(border=True):
    with st.expander("📋 Select Stocks and Run Analysis", expanded=True):
        with st.form(key='user_input_form'):
            st.markdown("### Select Stocks and Date Range", help="Choose stocks and historical period for analysis")
            
            # Handle the list to prevent errors if it's empty
            default_selection = [ticker_to_name.get(t) for t in st.session_state['custom_analysis_tickers']]
            
            selected_names = st.multiselect(
                "Select Stocks",
                options=master_names,
                default=default_selection
            )
            selected_tickers = [name_to_ticker.get(name) for name in selected_names if name_to_ticker.get(name)]

            today = dt.date.today()
            default_start_date = today - dt.timedelta(days=365)
            
            col1_form, col2_form = st.columns(2)
            with col1_form:
                start_date = st.date_input("Start date", default_start_date)
            with col2_form:
                end_date = st.date_input("End date", today)

            run_analysis = st.form_submit_button("🚀 Run Custom Analysis", use_container_width=True)
            
            if run_analysis:
                if start_date > end_date:
                    st.error("End date must be after start date.")
                    st.stop()
                if end_date > today:
                    st.error("The end date cannot be in the future. Please select a date on or before today.")
                    st.stop()
                
                st.session_state['custom_analysis_tickers'] = selected_tickers
                st.session_state['analysis_run'] = True
                
                # --- FIX: Force a data refresh for the new tickers before re-running ---
                # This prevents the KeyError by ensuring the master_data DataFrame is up-to-date
                st.session_state['data_refreshed_by_form'] = True
                st.rerun()
                
    if st.session_state['analysis_run'] and st.session_state['custom_analysis_tickers']:
        st.markdown("---") # Separator between the form and the results
        col1_qa, col2_qa = st.columns(2)

        custom_scores_df = compute_scores(st.session_state['master_data'], st.session_state['custom_analysis_tickers'])
        st.session_state['custom_scanner_df'] = custom_scores_df
        
        with col1_qa:
            st.markdown("#### Correlation Heatmap")
            
            # Handle the 'unhashable list' error and case for a single stock
            if len(st.session_state['custom_analysis_tickers']) > 1:
                try:
                    close_prices = st.session_state['master_data'].loc[:, pd.IndexSlice[st.session_state['custom_analysis_tickers'], 'Close']]
                    # Fix for potential multi-index issues
                    close_prices.columns = close_prices.columns.droplevel(1)
                    returns = close_prices.pct_change(fill_method=None).dropna()
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(returns.corr(), annot=False, cmap='viridis', ax=ax)
                    ax.set_title('Stock Correlation Heatmap')
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not generate heatmap: {e}. Please ensure selected stocks have enough data.")
            else:
                st.info("Select two or more stocks to display a correlation heatmap.")

            st.markdown("#### AI-Powered Summary")
            
            selected_stock_for_summary = st.selectbox(
                "Select a stock for AI summary",
                options=[ticker_to_name.get(t, t) for t in st.session_state['custom_analysis_tickers']],
                key='ai_summary_select'
            )

            if selected_stock_for_summary:
                selected_ticker = name_to_ticker.get(selected_stock_for_summary)
                
                # Ensure the stock exists in the scores DataFrame
                stock_scores_series = custom_scores_df[custom_scores_df['Ticker'] == selected_ticker]
                stock_scores = stock_scores_series.iloc[0] if not stock_scores_series.empty else None
                
                if stock_scores is not None:
                    # Create Radar Chart with Dark Theme
                    categories = ['MA50 Score', 'RSI Score', 'Volume Score', 'Final Score']
                    values = [
                        stock_scores['MA50 Score'],
                        stock_scores['RSI Score'],
                        stock_scores['Volume Score'],
                        stock_scores['Final Score']
                    ]
                    
                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name='Scores'
                    ))
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[-1, 1], # Scores are between -1 and 1
                                gridcolor='#444444',
                                linecolor='#666666',
                                tickfont=dict(color='#cccccc')
                            ),
                            angularaxis=dict(
                                gridcolor='#444444',
                                tickfont=dict(color='#cccccc')
                            )
                        ),
                        showlegend=False,
                        margin=dict(l=50, r=50, t=50, b=50),
                        height=400,
                        title=f"Custom Trend Score Breakdown for {selected_stock_for_summary}",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='#333333',
                        title_font_color='#ffffff'
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
                
                    with st.spinner(f"Generating AI analysis for {selected_stock_for_summary}..."):
                        prompt = f"""
                        You are a professional stock market analyst. Your task is to provide a brief, insightful summary of a single stock's recent performance.

                        The stock is {selected_stock_for_summary} ({selected_ticker}).
                        Its current custom trend score is {stock_scores['Final Score']:.2f} (where 1 is very bullish and -1 is very bearish).
                        The individual component scores are: MA50 Score: {stock_scores['MA50 Score']:.2f}, RSI Score: {stock_scores['RSI Score']:.2f}, Volume Score: {stock_scores['Volume Score']:.2f}.

                        Based on these scores and its recent performance, explain the stock's potential trend and market sentiment.

                        Please provide a response in JSON format only, with the following schema:
                        {{
                            "summary": "A brief, one-paragraph overview of the stock's performance in the context of its score.",
                            "key_points": [
                                "A bullet point describing the price action.",
                                "A bullet point on what the score implies.",
                                "A bullet point on potential next steps for the stock."
                            ],
                            "sentiment": "Overall sentiment, one of: 'Positive', 'Neutral', 'Negative'."
                        }}

                        Do not include any other text or markdown.
                        """
                        llm_data = get_llm_summary(prompt)
                    
                    if "error" in llm_data:
                        st.error(llm_data["error"])
                    else:
                        sentiment = llm_data.get('sentiment', 'N/A')
                        sentiment_class = sentiment.lower()
                        
                        st.markdown(f"#### Overall Sentiment: <span class='sentiment-{sentiment_class}'>{sentiment}</span>", unsafe_allow_html=True)
                        st.info(llm_data.get('summary', 'No summary provided.'))
                        st.markdown("##### Key Points")
                        for point in llm_data.get('key_points', []):
                            st.markdown(f"* {point}")
                else:
                    st.warning(f"Insufficient data to compute scores for {selected_stock_for_summary}. Please select a different stock or a longer time frame.")
            else:
                st.info("Select a stock from the dropdown above to get an AI summary.")

        with col2_qa:
            st.markdown("#### Final Scores (Custom Trend Scanner)")
            if not custom_scores_df.empty:
                st.dataframe(
                    custom_scores_df,
                    use_container_width=True,
                    column_config={
                        "Final Score": st.column_config.ProgressColumn("Final Score", help="Composite trend score", format="%.2f", min_value=-1, max_value=1)
                    }
                )
            else:
                st.info("No scores to display. Please re-run the analysis.")
            
            st.markdown("#### Raw Data Preview")
            
            selected_stock_for_raw = st.selectbox(
                "Select a stock for raw data",
                options=[ticker_to_name.get(t, t) for t in st.session_state['custom_analysis_tickers']],
                key='raw_data_select'
            )
            
            if selected_stock_for_raw:
                selected_ticker = name_to_ticker.get(selected_stock_for_raw)
                df_display = st.session_state['master_data'].loc[:, selected_ticker].dropna()
                
                if not df_display.empty:
                    st.dataframe(df_display.tail(10))
                else:
                    st.warning("No data found for the selected stock.")
            else:
                st.info("Select a stock from the dropdown above to view its data.")

            st.markdown("#### Technical Analysis Charts")
            if selected_stock_for_raw:
                selected_ticker = name_to_ticker.get(selected_stock_for_raw)
                df_chart = st.session_state['master_data'].loc[:, selected_ticker].dropna()
                
                if not df_chart.empty:
                    # Calculate indicators for the chart
                    df_chart['MA50'] = SMAIndicator(close=df_chart['Close'], window=50).sma_indicator()
                    df_chart['MA200'] = SMAIndicator(close=df_chart['Close'], window=200).sma_indicator()
                    df_chart['EMA200'] = EMAIndicator(close=df_chart['Close'], window=200).ema_indicator()
                    df_chart['RSI'] = RSIIndicator(close=df_chart['Close'], window=14).rsi()

                    # Candlestick chart
                    fig_candle = go.Figure(data=[go.Candlestick(
                        x=df_chart.index,
                        open=df_chart['Open'],
                        high=df_chart['High'],
                        low=df_chart['Low'],
                        close=df_chart['Close'],
                        name='Price'
                    )])
                    
                    # Add Moving Averages
                    fig_candle.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MA50'], mode='lines', name='MA50', line=dict(color='orange', width=2)))
                    fig_candle.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MA200'], mode='lines', name='MA200', line=dict(color='blue', width=2)))
                    fig_candle.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA200'], mode='lines', name='EMA200', line=dict(color='yellow', width=2, dash='dot')))
                    
                    fig_candle.update_layout(
                        title=f'{selected_stock_for_raw} Price Chart',
                        xaxis_rangeslider_visible=False,
                        height=500,
                        margin=dict(l=20, r=20, t=50, b=20),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='#333333',
                        font=dict(color='#cccccc'),
                        yaxis_title='Price (INR)'
                    )
                    st.plotly_chart(fig_candle, use_container_width=True)

                    # RSI chart
                    fig_rsi = go.Figure(data=[go.Scatter(x=df_chart.index, y=df_chart['RSI'], name='RSI', line=dict(color='#38b2ac', width=2))])
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought", annotation_position="bottom right")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold", annotation_position="top right")
                    fig_rsi.update_layout(
                        title=f'{selected_stock_for_raw} RSI',
                        height=200,
                        margin=dict(l=20, r=20, t=50, b=20),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='#333333',
                        font=dict(color='#cccccc'),
                        yaxis_title='RSI'
                    )
                    st.plotly_chart(fig_rsi, use_container_width=True)

                    # Volume chart
                    fig_volume = go.Figure(data=[go.Bar(
                        x=df_chart.index,
                        y=df_chart['Volume'],
                        name='Volume'
                    )])
                    fig_volume.update_layout(
                        title=f'{selected_stock_for_raw} Volume',
                        height=200,
                        margin=dict(l=20, r=20, t=50, b=20),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='#333333',
                        font=dict(color='#cccccc'),
                        yaxis_title='Volume'
                    )
                    st.plotly_chart(fig_volume, use_container_width=True)

                else:
                    st.info("No data found for the selected stock.")
            else:
                st.info("Select a stock from the dropdown above to see its charts.")
