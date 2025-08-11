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
import plotly.graph_objects as go
import datetime as dt
import json
import requests
import time

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
    
    .st-emotion-cache-16p6y9p {
        border-radius: 12px;
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
        else:
            st.warning(f"Data for {ticker} could not be downloaded.")
    
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values('Final Score', ascending=False).reset_index(drop=True)

# --- Function to call the LLM API and get JSON response ---
@st.cache_data(ttl=3600*4) # Cache the LLM response for 4 hours
def get_llm_summary(prompt, retry_count=3, backoff_factor=1.0):
    if not API_KEY:
        st.info("API key is not configured. The AI-powered summary will not be available.")
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
                default=[] 
            )
            selected_tickers = [name_to_ticker[name] for name in selected_names]

            today = dt.date.today()
            default_start_date = today - dt.timedelta(days=365)
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start date", default_start_date)
            with col2:
                end_date = st.date_input("End date", today)

            run_analysis = st.form_submit_button("🚀 Run Analysis", use_container_width=True)
            
            if run_analysis:
                if start_date > end_date:
                    st.error("End date must be after start date.")
                    st.stop()
                if end_date > today:
                    st.error("The end date cannot be in the future. Please select a date on or before today.")
                    st.stop()
    
    if run_analysis and not selected_tickers:
        st.warning("Please select at least one stock.")
        st.stop()
    elif run_analysis and selected_tickers:
        data = get_stock_data(selected_tickers, start_date, end_date)
        
        if data.empty:
            st.error("No data available for selected period.")
            st.stop()
        
        scanner_df = compute_scores(data, selected_tickers)
        
        if scanner_df.empty:
            st.warning("Not enough data for analysis (need min 200 days).")
            st.stop()
        
        st.session_state['scanner_df'] = scanner_df
        st.session_state['selected_tickers'] = selected_tickers
        st.session_state['data'] = data
    
if 'scanner_df' in st.session_state and not st.session_state['scanner_df'].empty:
    scanner_df = st.session_state['scanner_df']
    selected_tickers = st.session_state['selected_tickers']
    data = st.session_state['data']

    with left_column:
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

        # Raw Data Preview
        with st.expander("🔍 View Raw Data"):
            selected_raw_ticker = st.selectbox(
                "Select stock:",
                options=selected_tickers,
                format_func=lambda x: ticker_to_name[x]
            )
            if selected_raw_ticker in data.columns:
                st.dataframe(data[selected_raw_ticker].tail(10))


    # --- Right Column Content ---
    with right_column:

        # LLM Qualitative Analysis with Visuals
        st.markdown("## 💬 Qualitative Analysis")
        with st.expander("AI-powered Summary", expanded=True):
            if 'llm_data' not in st.session_state:
                st.session_state['llm_data'] = None
                
            selected_name_for_summary = st.selectbox(
                "Select a stock to get a qualitative summary:",
                options=scanner_df['Company Name'].tolist(),
                key='summary_selectbox'
            )
            
            if st.button("Generate Summary", use_container_width=True):
                selected_row = scanner_df[scanner_df['Company Name'] == selected_name_for_summary].iloc[0]
                
                prompt = f"""
                You are a professional stock market analyst. Your task is to provide a brief, insightful summary of a stock's recent performance based on its technical scores.

                Please provide a response in JSON format only, with the following schema:
                {{
                    "summary": "A brief, one-paragraph overview of the stock's performance.",
                    "key_points": [
                        "A bullet point describing the MA50 trend.",
                        "A bullet point describing the RSI momentum.",
                        "A bullet point describing the volume activity."
                    ],
                    "sentiment": "Overall sentiment, one of: 'Positive', 'Neutral', 'Negative'."
                }}

                Here are the scores for {selected_name_for_summary}:
                - Final Score (composite): {selected_row['Final Score']:.2f} (from -1 to 1)
                - MA50 Score (trend): {selected_row['MA50 Score']:.2f} (from -1 to 1)
                - RSI Score (momentum): {selected_row['RSI Score']:.2f} (from -1 to 1)
                - Volume Score (volume strength): {selected_row['Volume Score']:.2f} (from -1 to 1)

                Do not include any other text or markdown.
                """
                with st.spinner("Generating summary and visuals..."):
                    llm_data = get_llm_summary(prompt)
                    st.session_state['llm_data'] = llm_data
            
            if 'llm_data' in st.session_state and st.session_state['llm_data']:
                llm_data = st.session_state['llm_data']
                if "error" in llm_data:
                    st.error(llm_data["error"])
                else:
                    # Create a radar chart to visualize the scores
                    categories = ['MA50 Score', 'RSI Score', 'Volume Score', 'Final Score']
                    
                    # Get actual scores for the chart
                    selected_row = scanner_df[scanner_df['Company Name'] == selected_name_for_summary].iloc[0]
                    scores_for_chart = [
                        selected_row['MA50 Score'],
                        selected_row['RSI Score'],
                        selected_row['Volume Score'],
                        selected_row['Final Score']
                    ]

                    fig = go.Figure()

                    fig.add_trace(go.Scatterpolar(
                        r=scores_for_chart,
                        theta=categories,
                        fill='toself',
                        name='Technical Scores',
                        line=dict(color='#38b2ac')
                    ))

                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[-1, 1],
                                showticklabels=False
                            )),
                        showlegend=False,
                        height=400,
                        margin=dict(l=50, r=50, t=50, b=50),
                        template='plotly_dark'
                    )

                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown(f"### Overall Sentiment: {llm_data.get('sentiment', 'N/A')}")
                    st.info(llm_data.get('summary', 'No summary provided.'))
                    st.markdown("#### Key Points")
                    for point in llm_data.get('key_points', []):
                        st.markdown(f"* {point}")

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

