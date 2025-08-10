#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import talib as ta
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import datetime as dt

# --- Streamlit App Configuration (with wide layout) and CSS ---
st.set_page_config(layout="wide", page_title="Stock Scanner")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap');
    
    html, body, [class*="st-emotion-cache"] {
        font-family: 'Inter', sans-serif;
        color: #f0f0f0;
    }
    
    .main {
        background-color: #1a1a1a;
    }

    .block-container {
        padding: 2rem !important;
        max-width: 100% !important;
    }

    h1, h2, h3, h4 {
        color: #ffffff;
    }
    
    /* Custom CSS for the container cards */
    .st-emotion-cache-1wivd2v { 
        background-color: #2a2a2a;
        border: 1px solid #444444;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    .st-emotion-cache-1f81kof {
        border-radius: 12px;
    }
    
    /* Main Dashboard Title and Subtitle */
    .dashboard-title {
        font-size: 3rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0;
    }
    .dashboard-subtitle {
        font-size: 1.25rem;
        font-weight: 400;
        color: #cccccc;
        margin-top: 0;
    }
            

    /* Colors for scores - Adjusted for contrast */
    .positive-score {
        color: #38b2ac; /* A teal-like green, visually distinct */
        font-weight: 700;
        font-size: 1.1rem;
        margin-right: auto;
    }
    .negative-score {
        color: #e53e3e; /* A strong red */
        font-weight: 700;
        font-size: 1.1rem;
    }
    .neutral-score {
        color: #f6ad55; /* An orange */
        font-weight: 700;
        font-size: 1.1rem;
    }
            

   
     /* New CSS to make the main dashboard cards more square */
    .dashboard-performer-card {
        background-color: #2a2a2a;
        border: 1px solid #444444;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        min-height: 250px; /* Adjust this value to control the height */
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
    }
    

    /* Buttons and Input Styling */
    .stButton>button {
        border-radius: 8px;
        border: 1px solid #444444;
        background-color: #333333;
        color: #f0f0f0;
    }
    .stButton>button:hover {
        border-color: #007bff;
        color: #007bff;
    }
    .st-emotion-cache-1c99r31, .st-emotion-cache-1n1f13 {
        background-color: #333333;
        color: #f0f0f0;
        border-color: #444444;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="dashboard-title">Enhanced Stock Scanner</p>', unsafe_allow_html=True)
st.markdown('<p class="dashboard-subtitle">Interactive dashboard to analyze a custom selection of stocks using technical indicators.</p>', unsafe_allow_html=True)

# --- Master Tickers List & Company Names Mapping ---
# Dow 30 Tickers list is defined here
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
    "ONGC.NS", "POWERGRID.NS", "Power Grid Corporation", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS",
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
@st.cache_data(ttl=3600)  # Cache data for 1 hour to prevent excessive API calls
def get_stock_data(tickers, start_date, end_date):
    """
    Downloads daily data for a list of tickers and caches it.
    Includes robust error handling for network issues.
    """
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
        st.error(f"Failed to download data from Yahoo Finance. Please check your network connection and try again.")
        st.error(f"Error details: {e}")
        return pd.DataFrame()

# --- Indicator Functions ---
def ma50_score(df):
    if len(df) < 200:
        return 0.0
    ma50_series = ta.SMA(df['Close'], timeperiod=50)
    ma200_series = ta.SMA(df['Close'], timeperiod=200)
    if ma50_series.isnull().iloc[-1] or ma200_series.isnull().iloc[-1]:
        return 0.0
    ma50 = ma50_series.iloc[-1]
    ma50_5ago = ma50_series.iloc[-5] if len(ma50_series) >= 5 else ma50
    ma200 = ma200_series.iloc[-1]
    close = df['Close'].iloc[-1]
    if ma50 == 0: dist_score = 0
    else: dist = (close - ma50) / ma50
    dist_score = max(-1, min(1, dist * 5))
    if ma50_5ago == 0: slope_score = 0
    else: slope = (ma50 - ma50_5ago) / ma50_5ago
    slope_score = max(-1, min(1, slope * 100))
    regime = 1 if ma50 > ma200 else -1
    score = 0.5 * dist_score + 0.3 * slope_score + 0.2 * regime
    return max(-1, min(1, score))

def rsi_score_momentum(df, rsi_period=14, lookback=20):
    if len(df) < rsi_period + lookback:
        return 0.0
    rsi_series = ta.RSI(df['Close'], timeperiod=rsi_period)
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

# --- Compute Scores Function (cached) ---
@st.cache_data
def compute_scores(data_frame, selected_tickers):
    results = []
    for ticker in selected_tickers:
        # Check if the ticker exists in the DataFrame
        if ticker in data_frame.columns.get_level_values(0):
            df_ticker = data_frame[ticker].dropna()
            if not df_ticker.empty and len(df_ticker) > 200:
                ma50_s = ma50_score(df_ticker)
                rsi_s = rsi_score_momentum(df_ticker)
                vol_s = vol_score(df_ticker)
                result = {
                    'Ticker': ticker,
                    'Company Name': ticker_to_name.get(ticker, ticker), # Use full name
                    'MA50 Score': ma50_s,
                    'RSI Score': rsi_s,
                    'Volume Score': vol_s,
                }
                result['Final Score'] = (result['MA50 Score'] + result['RSI Score'] + result['Volume Score']) / 3
                results.append(result)
        else:
            st.warning(f"Data for {ticker} could not be downloaded. It will be excluded from the analysis.")

    if not results:
        return pd.DataFrame()
    scanner_df = pd.DataFrame(results)
    return scanner_df.sort_values('Final Score', ascending=False).reset_index(drop=True)

# --- Main Dashboard Layout ---
left_column, right_column = st.columns([1, 1])

# --- Date Range and Ticker Selection (in a single collapsible expander) ---
with left_column:
    with st.expander("Expand to Change Inputs", expanded=True):
        with st.form(key='user_input_form'):
            st.subheader("Select Stocks and Date Range")
            st.markdown("Choose the stocks to analyze and the historical period.")

            # Dropdown for stock selection (displays names, returns tickers)
            selected_names = st.multiselect(
                "Select Stocks",
                options=master_names,
                default=[ticker_to_name[t] for t in dow30_tickers] # Default to Dow 30 names
            )
            selected_tickers = [name_to_ticker[name] for name in selected_names]

            today = dt.date.today()
            # Set the default start date to one year ago to ensure enough data points
            default_start_date = today - dt.timedelta(days=365)

            start_date = st.date_input("Start date", default_start_date)
            end_date = st.date_input("End date", today)

            if start_date > end_date:
                st.error("Error: The end date must be after the start date.")
                st.stop()
            
            # Form submit button
            submit_button = st.form_submit_button("Run Analysis")

    if not selected_tickers:
        st.warning("Please select at least one stock to analyze.")
        st.stop()

    # Fetch data after the form is submitted or on initial load
    data = get_stock_data(selected_tickers, start_date, end_date)
    
    # Check if data download was successful and not empty
    if data.empty:
        st.stop()

    scanner_df = compute_scores(data, selected_tickers)
    
    if scanner_df.empty:
        st.warning("Not enough data to calculate scores for the selected period. Please choose a longer date range (at least 200 days).")
        st.stop()
    
    st.header("Top & Bottom Performers")
    st.markdown("A quick visual summary of the highest and lowest scoring stocks.")
    
    top_bottom_col1, top_bottom_col2 = st.columns(2)

    with top_bottom_col1:
        st.subheader("Top Performers 📈")
        top_3 = scanner_df.head(3)
        if not top_3.empty:
            for _, row in top_3.iterrows():
                # Corrected rendering of company name and score
                with st.container(border=True):
                    st.markdown(f"**{row['Company Name']}** <span class='positive-score'>{row['Final Score']:.2f}</span>", unsafe_allow_html=True)
        else:
            st.info("Not enough data to show top performers.")

    with top_bottom_col2:
        st.subheader("Bottom Performers 📉")
        bottom_3 = scanner_df.tail(3)
        if not bottom_3.empty:
            for _, row in bottom_3.iterrows():
                with st.container(border=True):
                    # Corrected rendering of company name and score
                    st.markdown(f"**{row['Company Name']}** <span class='negative-score'>{row['Final Score']:.2f}</span>", unsafe_allow_html=True)
        else:
            st.info("Not enough data to show bottom performers.")

    # --- HEATMAP ---
    st.header("Stock Scanner Heatmap")
    with st.container(border=True):
        st.markdown("""
        This heatmap visualizes the final scores for all stocks in the scanner, providing a quick overview of the market sentiment.
        """)
        
        scores = scanner_df.copy().reset_index(drop=True)
        values = scores['Final Score'].values

        if len(values) == 0:
            st.warning("No data to display in the heatmap.")
            st.stop()

        # Determine grid size (approx square)
        n = len(values)
        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))

        grid = np.full((rows, cols), np.nan)
        labels = np.full((rows, cols), "", dtype=object)

        for i, val in enumerate(values):
            r = i // cols
            c = i % cols
            grid[r, c] = val
            labels[r, c] = scores['Company Name'].iloc[i] # Use company name here

        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#2a2a2a')
        sns.heatmap(grid,
                    annot=labels,
                    fmt='',
                    center=0,
                    cmap='RdYlGn',
                    cbar_kws={'label': 'Final Score'},
                    linewidths=0.5,
                    linecolor='gray',
                    ax=ax)
        ax.set_title('Stock Scanner Heatmap', fontsize=16, color='white')
        ax.set_yticks([])
        ax.set_xticks([])
        
        # Customize the colorbar
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.label.set_color('white')
        cbar.ax.tick_params(colors='white')
        
        st.pyplot(fig)


# --- Content for the Right Column ---
with right_column:
    if data.empty:
        st.stop()
        
    scanner_df = compute_scores(data, selected_tickers)

    if scanner_df.empty:
        st.warning("Not enough data to calculate scores for the selected period. Please choose a longer date range (at least 200 days).")
        st.stop()
        
    st.header("Scanner Results Table")
    st.markdown("The table below shows the calculated scores for each stock, sorted by the `Final Score`.")
    st.dataframe(scanner_df.style.bar(subset=['Final Score'], align='mid', color=['#e54b4b', '#1a9953']), use_container_width=True)

    st.subheader("Final Score per Ticker")
    fig = px.bar(
        scanner_df,
        x='Company Name', # Use company name on the x-axis
        y='Final Score',
        color=np.where(scanner_df['Final Score'] > 0, 'Positive', 'Negative'),
        color_discrete_map={'Positive': '#1a9953', 'Negative': '#e54b4b'},
        labels={'Final Score': 'Combined Technical Score'},
        title='Technical Score for Each Ticker',
        height=500,
        template="plotly_dark"
    )
    fig.update_layout(xaxis={'categoryorder':'total descending'})
    st.plotly_chart(fig, use_container_width=True)

    st.header("Raw Data Preview")
    with st.expander("Show raw data for a selected stock"):
        selected_raw_ticker = st.selectbox("Select a stock to view its raw data:", options=selected_tickers, format_func=lambda x: ticker_to_name[x])
        if selected_raw_ticker and selected_raw_ticker in data.columns:
            st.markdown(f"Here's a look at the raw data for **{ticker_to_name[selected_raw_ticker]} ({selected_raw_ticker})**.")
            st.dataframe(data[selected_raw_ticker].tail())
        else:
            st.info("No data available for the selected stock.")