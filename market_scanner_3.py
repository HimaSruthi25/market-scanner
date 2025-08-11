#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf 
import ta
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from ta.volume import VolumeWeightedAveragePrice
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import datetime as dt

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
    
    /* Card styling */
    .st-emotion-cache-1wivd2v {
        background-color: #2a2a2a;
        border: 1px solid #444444;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        margin-top: 1rem !important;
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
            
</style>
""", unsafe_allow_html=True)

# --- App Header ---
st.markdown("""
<div class="dashboard-title-container">
    <h1 class="dashboard-title">Enhanced Stock Scanner</h1>
    <p class="dashboard-subtitle">Interactive dashboard to analyze a custom selection of stocks using technical indicators.</p>
</div>
""", unsafe_allow_html=True)

# --- Master Tickers List & Company Names Mapping ---
dow30_tickers = [
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW",
    "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
    "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT"
]

nifty50_tickers = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BHARTIARTL.NS", "BPCL.NS",
    "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS",
    "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS",
    "INFY.NS", "ITC.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS",
    "LTIM.NS", "M&M.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS",
    "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS",
    "SUNPHARMA.NS", "TCS.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS",
    "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "ZEEL.NS"
]

ticker_to_name = {
    # Dow 30 Tickers
    "AAPL": "Apple", "AMGN": "Amgen", "AXP": "American Express", "BA": "Boeing",
    "CAT": "Caterpillar", "CRM": "Salesforce", "CSCO": "Cisco", "CVX": "Chevron",
    "DIS": "Disney", "DOW": "Dow Inc", "GS": "Goldman Sachs", "HD": "Home Depot",
    "HON": "Honeywell", "IBM": "IBM", "INTC": "Intel", "JNJ": "Johnson & Johnson",
    "JPM": "JPMorgan Chase", "KO": "Coca-Cola", "MCD": "McDonald's", "MMM": "3M",
    "MRK": "Merck", "MSFT": "Microsoft", "NKE": "Nike", "PG": "Procter & Gamble",
    "TRV": "Travelers", "UNH": "UnitedHealth", "V": "Visa", "VZ": "Verizon",
    "WBA": "Walgreens Boots Alliance", "WMT": "Walmart",
    # Nifty 50 Tickers
    "ADANIENT.NS": "Adani Enterprises", "ADANIPORTS.NS": "Adani Ports", "APOLLOHOSP.NS": "Apollo Hospitals",
    "ASIANPAINT.NS": "Asian Paints", "AXISBANK.NS": "Axis Bank", "BAJAJ-AUTO.NS": "Bajaj Auto",
    "BAJFINANCE.NS": "Bajaj Finance", "BAJAJFINSV.NS": "Bajaj Finserv", "BHARTIARTL.NS": "Bharti Airtel",
    "BPCL.NS": "BPCL", "BRITANNIA.NS": "Britannia", "CIPLA.NS": "Cipla",
    "COALINDIA.NS": "Coal India", "DIVISLAB.NS": "Divi's Laboratories", "DRREDDY.NS": "Dr Reddy's Laboratories",
    "EICHERMOT.NS": "Eicher Motors", "GRASIM.NS": "Grasim Industries", "HCLTECH.NS": "HCL Technologies",
    "HDFCBANK.NS": "HDFC Bank", "HDFCLIFE.NS": "HDFC Life Insurance", "HEROMOTOCO.NS": "Hero MotoCorp",
    "HINDALCO.NS": "Hindalco", "HINDUNILVR.NS": "Hindustan Unilever", "ICICIBANK.NS": "ICICI Bank",
    "INDUSINDBK.NS": "IndusInd Bank", "INFY.NS": "Infosys", "ITC.NS": "ITC Ltd",
    "JSWSTEEL.NS": "JSW Steel", "KOTAKBANK.NS": "Kotak Mahindra Bank", "LT.NS": "Larsen & Toubro",
    "LTIM.NS": "LTI Mindtree", "M&M.NS": "Mahindra & Mahindra", "MARUTI.NS": "Maruti Suzuki",
    "NESTLEIND.NS": "Nestle India", "NTPC.NS": "NTPC", "ONGC.NS": "ONGC",
    "POWERGRID.NS": "Power Grid Corporation", "RELIANCE.NS": "Reliance Industries", "SBILIFE.NS": "SBI Life Insurance",
    "SBIN.NS": "State Bank of India", "SUNPHARMA.NS": "Sun Pharma", "TCS.NS": "TCS",
    "TATACONSUM.NS": "Tata Consumer Products", "TATAMOTORS.NS": "Tata Motors", "TATASTEEL.NS": "Tata Steel",
    "TECHM.NS": "Tech Mahindra", "TITAN.NS": "Titan", "ULTRACEMCO.NS": "UltraTech Cement",
    "WIPRO.NS": "Wipro", "ZEEL.NS": "Zee Entertainment"
}

master_tickers = sorted(ticker_to_name.keys())
master_names = sorted(ticker_to_name.values())
name_to_ticker = {v: k for k, v in ticker_to_name.items()}

# --- Data Download Function (cached) ---
@st.cache_data(ttl=3600)
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

# --- Indicator Functions ---
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
    if ma50 == 0: 
        dist_score = 0
    else: 
        dist = (close - ma50) / ma50 
    dist_score = max(-1, min(1, dist * 5))
    if ma50_5ago == 0: 
        slope_score = 0
    else: 
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

# --- Compute Scores Function ---
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
        else:
            st.warning(f"Data for {ticker} could not be downloaded.")
    
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values('Final Score', ascending=False).reset_index(drop=True)

# --- Main Dashboard Layout ---
left_column, right_column = st.columns([1, 1])

# --- Left Column Content ---
with left_column:
    with st.expander(" 📋 Search ", expanded=True):
        with st.form(key='user_input_form'):
            st.markdown("### Select Stocks and Date Range", help="Choose stocks and historical period for analysis")
            
            selected_names = st.multiselect(
                "Select Stocks",
                options=master_names,
                default=[]  # Default first 10 Dow stocks
            )
            selected_tickers = [name_to_ticker[name] for name in selected_names]

            today = dt.date.today()
            default_start_date = today - dt.timedelta(days=365)
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start date", default_start_date)
            with col2:
                end_date = st.date_input("End date", today)

            if st.form_submit_button("🚀 Run Analysis", use_container_width=True):
                if start_date > end_date:
                    st.error("End date must be after start date")
                    st.stop()

    if not selected_tickers:
        st.warning("Please select at least one stock")
        st.stop()

    data = get_stock_data(selected_tickers, start_date, end_date)
    
    if data.empty:
        st.error("No data available for selected period")
        st.stop()

    scanner_df = compute_scores(data, selected_tickers)
    
    if scanner_df.empty:
        st.warning("Not enough data for analysis (need min 200 days)")
        st.stop()
    
    # Top & Bottom Performers
    st.markdown("## 📊 Top & Bottom Performers", help="Highest and lowest scoring stocks")
    
    top_col, bottom_col = st.columns(2)
    with top_col:
        st.markdown("### 🔺 Top 3")
        top_3 = scanner_df.head(3)
        if not top_3.empty:
            for _, row in top_3.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{row['Company Name']}** <span class='positive-score'>{row['Final Score']:.2f}</span>", 
                               unsafe_allow_html=True)
    
    with bottom_col:
        st.markdown("### 🔻 Bottom 3")
        bottom_3 = scanner_df.tail(3)
        if not bottom_3.empty:
            for _, row in bottom_3.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{row['Company Name']}** <span class='negative-score'>{row['Final Score']:.2f}</span>", 
                               unsafe_allow_html=True)

    # Heatmap Visualization
    st.markdown("## 🔥 Stock Scanner Heatmap")
    with st.container(border=True):
        scores = scanner_df.copy()
        values = scores['Final Score'].values
        
        if len(values) > 0:
            n = len(values)
            rows = int(np.ceil(np.sqrt(n)))
            cols = int(np.ceil(n / rows))
            grid = np.full((rows, cols), np.nan)
            labels = np.full((rows, cols), "", dtype=object)

            for i, val in enumerate(values):
                r = i // cols
                c = i % cols
                grid[r, c] = val
                labels[r, c] = scores['Company Name'].iloc[i]

            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(grid, annot=labels, fmt='', center=0, cmap='RdYlGn',
                        cbar_kws={'label': 'Final Score'}, linewidths=0.5, linecolor='gray', ax=ax)
            ax.set_title('Stock Scanner Heatmap', fontsize=16)
            ax.set_xticks([])
            ax.set_yticks([])
            st.pyplot(fig)
        else:
            st.warning("No data available for heatmap")

# --- Right Column Content ---
with right_column:
    if not scanner_df.empty:
        # Results Table
        st.markdown("## 📋 Scanner Results")
        st.dataframe(
            scanner_df.style.format({
                'MA50 Score': '{:.2f}',
                'RSI Score': '{:.2f}',
                'Volume Score': '{:.2f}',
                'Final Score': '{:.2f}'
            }).bar(subset=['Final Score'], color=['#e53e3e', '#38b2ac']),
            use_container_width=True,
            hide_index=True,
            height=min(500, 60 + len(scanner_df) * 35)
        )

        # Final Score Bar Chart
        st.markdown("## 📈 Final Scores")
        fig = px.bar(
            scanner_df,
            x='Company Name',
            y='Final Score',
            color=np.where(scanner_df['Final Score'] > 0, 'Positive', 'Negative'),
            color_discrete_map={'Positive': '#38b2ac', 'Negative': '#e53e3e'},
            labels={'Final Score': 'Score'},
            height=500
        )
        fig.update_layout(xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig, use_container_width=True)

        # Raw Data Preview
        with st.expander("🔍 View Raw Data"):
            selected_raw_ticker = st.selectbox(
                "Select stock:",
                options=selected_tickers,
                format_func=lambda x: ticker_to_name[x]
            )
            if selected_raw_ticker in data.columns:


                st.dataframe(data[selected_raw_ticker].tail(10))
