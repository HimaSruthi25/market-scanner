#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import time
import pytz
import datetime as dt
from pathlib import Path

# plotting & TA
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import ta
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator
from ta.volume import VolumeWeightedAveragePrice

# yfinance import kept but NOT used as fallback per your request (so app won't call it)
import yfinance as yf  # retained for compatibility but not used

# TradingView embed
import streamlit.components.v1 as components

# Zerodha Kite Connect imports
from kiteconnect import KiteConnect

# -----------------------
# API KEYS / ENDPOINTS
# -----------------------
# Gemini (unchanged)
# Gemini keys
API_KEY = st.secrets["gemini"]["api_key"]
GEMINI_API_URL = st.secrets["gemini"]["api_url"]


# Zerodha API credentials and redirect URL
Z_API_KEY = st.secrets["zerodha"]["api_key"]
Z_API_SECRET = st.secrets["zerodha"]["api_secret"]
Z_REDIRECT_URL = st.secrets["zerodha"]["redirect_url"]


# Local cache path
CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)
MASTER_CACHE_PATH = CACHE_DIR / "last_master_data.parquet"

# --- Streamlit App Configuration with Layout Fixes ---
st.set_page_config(
    layout="wide",
    page_title="Stock Scanner with Zerodha",
    initial_sidebar_state="expanded"
)

# --- CSS (unchanged from your UI) ---
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

# --- Zerodha Kite Connect Authentication Section --- (Insert near app start)

kite = KiteConnect(api_key=Z_API_KEY)
query_params = st.query_params
request_token = query_params.get("request_token", [None])[0]

if request_token:
    try:
        data = kite.generate_session(request_token, api_secret=Z_API_SECRET)
        access_token = data["access_token"]
        kite.set_access_token(access_token)
        st.session_state['kite_access_token'] = access_token
        st.success("Successfully authenticated with Zerodha!")
    except Exception as e:
        st.error(f"Authentication Error: {e}")
        st.stop()
elif 'kite_access_token' in st.session_state:
    kite.set_access_token(st.session_state['kite_access_token'])
else:
    login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={Z_API_KEY}&redirect_url={Z_REDIRECT_URL}"
    st.markdown(f"[Login with Zerodha's Kite Connect]({login_url}) to start")
    st.stop()

# --- Fetch Nifty 100 Stock List (unchanged) ---
@st.cache_data(ttl=3600 * 6)
def get_nifty_indices_data():
    nifty100_url = "https://archives.nseindia.com/content/indices/ind_nifty100list.csv"
    ticker_to_name = {}
    nifty100_tickers = []
    try:
        nifty100_df = pd.read_csv(nifty100_url)
        nifty100_tickers = [symbol + '.NS' for symbol in nifty100_df['Symbol'].tolist()]
        for _, row in nifty100_df.iterrows():
            ticker_to_name[row['Symbol'] + '.NS'] = row['Company Name']
    except Exception as e:
        print(f"[WARN] Failed to fetch NSE list: {e}")
        return [], [], {}
    return nifty100_tickers, nifty100_tickers, ticker_to_name

with st.spinner("Fetching the latest Nifty 100 stock list..."):
    nifty50_tickers, nifty100_tickers, ticker_to_name = get_nifty_indices_data()

master_tickers = sorted(list(set(nifty100_tickers)))
master_names = sorted([ticker_to_name.get(t, t.split('.')[0]) for t in master_tickers])
name_to_ticker = {v: k for k, v in ticker_to_name.items()}

# --- Zerodha Instruments Cache ---
@st.cache_data(ttl=86400)
def get_nse_instruments():
    try:
        return kite.instruments("NSE")
    except Exception as e:
        st.error(f"Error fetching Zerodha instruments: {e}")
        return []

instruments = get_nse_instruments()
symbol_to_token = {inst['tradingsymbol']: inst['instrument_token'] for inst in instruments}

# --- Replace IndianAPI Fetch with Zerodha Kite Connect Fetch ---
@st.cache_data(ttl=300)
def get_stock_data(tickers, start_date, end_date):
    per_ticker = {}
    for ticker in tickers:
        symbol = ticker.replace(".NS", "")  # Convert ticker to Zerodha symbol
        token = symbol_to_token.get(symbol)
        if not token:
            st.warning(f"No instrument token found for {ticker}")
            continue
        try:
            data = kite.historical_data(token, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), "day")
            if not data:
                continue
            df = pd.DataFrame(data)
            df.rename(columns={
                'date': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
            }, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            per_ticker[ticker] = df
        except Exception as e:
            st.warning(f"Failed fetching data for {ticker}: {e}")
            continue
    if per_ticker:
        return pd.concat(per_ticker, axis=1)
    return pd.DataFrame()

# --- Centralized data fetching logic with Zeodhra Kite Connect ---
all_tickers_to_fetch = list(set(nifty100_tickers + st.session_state.get('custom_analysis_tickers', [])))
master_data = pd.DataFrame()
force_data_refresh = False


# --- Shortcut to simulate market open and force scanning on cached data ---
simulate_market_open = st.sidebar.checkbox("Simulate Market Open for Testing", value=False)

def is_market_open():
    if simulate_market_open:
        return True
    ist = pytz.timezone('Asia/Kolkata')
    now = dt.datetime.now(ist)
    start_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
    end_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
    is_weekday = now.weekday() < 5
    is_open = now >= start_time and now <= end_time
    return is_weekday and is_open

if is_market_open():
    if (dt.datetime.now() - st.session_state.get('last_run_time', dt.datetime.min)).total_seconds() > 300:  # 5 minutes
        force_data_refresh = True
        st.info("🔄 Market is open. Refreshing live data...")
    else:
        master_data = st.session_state.get('master_data', pd.DataFrame())
else:
    if st.session_state.get('master_data', pd.DataFrame()).empty:
        force_data_refresh = True
        st.info("Market is closed. Fetching end-of-day data.")
    else:
        master_data = st.session_state.get('master_data', pd.DataFrame())

if force_data_refresh or 'data_refreshed_by_form' in st.session_state:
    if 'data_refreshed_by_form' in st.session_state:
        del st.session_state['data_refreshed_by_form'] # Clear flag
    start_date = dt.date.today() - dt.timedelta(days=365)
    end_date = dt.date.today()
    try:
        fetched = get_stock_data(all_tickers_to_fetch, start_date, end_date)
        if fetched is not None and not fetched.empty:
            master_data = fetched
            st.session_state['master_data'] = master_data
            st.session_state['last_run_time'] = dt.datetime.now()
        else:
            st.warning("No data returned from Zerodha API.")
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")

if master_data.empty:
    st.warning("No stock data available. Please check Zerodha API connection.")
    st.stop()

# --- Scoring Functions (unchanged, use `master_data` as input) ---
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
    if data_frame is None or data_frame.empty:
        return pd.DataFrame()
    for ticker in selected_tickers:
        try:
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
        except Exception as e:
            print(f"[WARN] compute_scores skipped {ticker} due to error: {e}")
            continue

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values('Final Score', ascending=False).reset_index(drop=True)

# --- Scanner Functions (unchanged) ---
def find_52_week_high(df_all):
    results = []
    if df_all is None or df_all.empty:
        return results
    for ticker in df_all.columns.get_level_values(0).unique():
        try:
            df = df_all[ticker].dropna()
            if len(df) >= 252:
                high_52_weeks = df['High'].iloc[-252:].max()
                if df['Close'].iloc[-1] >= high_52_weeks:
                    results.append(ticker)
        except Exception:
            continue
    return results

def find_open_equals_high_low(df_all):
    results = []
    if df_all is None or df_all.empty:
        return results
    for ticker in df_all.columns.get_level_values(0).unique():
        try:
            df = df_all[ticker].dropna()
            if not df.empty:
                open_price = df['Open'].iloc[-1]
                high_price = df['High'].iloc[-1]
                low_price = df['Low'].iloc[-1]
                if open_price == high_price or open_price == low_price:
                    results.append(ticker)
        except Exception:
            continue
    return results

def find_bullish_engulfing(df_all):
    results = []
    if df_all is None or df_all.empty:
        return results
    for ticker in df_all.columns.get_level_values(0).unique():
        try:
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
        except Exception:
            continue
    return results

def find_volume_breaker(df_all, multiplier=2, window=20):
    results = []
    if df_all is None or df_all.empty:
        return results
    for ticker in df_all.columns.get_level_values(0).unique():
        try:
            df = df_all[ticker].dropna()
            if len(df) > window:
                current_volume = df['Volume'].iloc[-1]
                average_volume = df['Volume'].iloc[-window-1:-1].mean()
                if average_volume > 0 and current_volume > (average_volume * multiplier):
                    results.append(ticker)
        except Exception:
            continue
    return results

def find_200_ema_crossover(df_all):
    results = []
    if df_all is None or df_all.empty:
        return results
    for ticker in df_all.columns.get_level_values(0).unique():
        try:
            df = df_all[ticker].dropna()
            if len(df) >= 200:
                df['EMA200'] = EMAIndicator(close=df['Close'], window=200).ema_indicator()
                if (df['Close'].iloc[-1] > df['EMA200'].iloc[-1] and
                    df['Close'].iloc[-2] < df['EMA200'].iloc[-2]):
                    results.append(ticker)
        except Exception:
            continue
    return results

def find_golden_cross(df_all):
    results = []
    if df_all is None or df_all.empty:
        return results
    for ticker in df_all.columns.get_level_values(0).unique():
        try:
            df = df_all[ticker].dropna()
            if len(df) >= 200:
                ma50 = SMAIndicator(close=df['Close'], window=50).sma_indicator()
                ma200 = SMAIndicator(close=df['Close'], window=200).sma_indicator()
                if len(ma50) > 1 and len(ma200) > 1:
                    if ma50.iloc[-1] > ma200.iloc[-1] and ma50.iloc[-2] <= ma200.iloc[-2]:
                        results.append(ticker)
        except Exception:
            continue
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

