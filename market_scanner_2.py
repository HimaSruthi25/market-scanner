import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import datetime as dt
from datetime import timedelta

# ==============================================
# PAGE CONFIGURATION
# ==============================================
st.set_page_config(
    page_title='NSE Pulse Pro',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Force dark theme
st._config.set_option('theme.base', 'dark')
st._config.set_option('theme.primaryColor', '#4e79a7')
st._config.set_option('theme.backgroundColor', '#0e1117')
st._config.set_option('theme.secondaryBackgroundColor', '#1a1c23')
st._config.set_option('theme.textColor', '#fafafa')

# Custom CSS for dark theme
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #0e1117;
    }
    
    /* Metric card styling */
    div[data-testid="metric-container"] {
        background-color: #1a1c23;
        border-radius: 8px;
        padding: 15px 10px;
        margin: 5px;
        border-left: 4px solid #4e79a7;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
        padding: 0 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        padding: 0 20px;
        border-radius: 8px 8px 0 0;
        background-color: #1a1c23;
        transition: all 0.2s;
        font-weight: 500;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #252730;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4e79a7;
        color: white;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 8px;
        background-color: #1a1c23;
    }
    
    /* Table header styling */
    thead tr th {
        background-color: #1a1c23 !important;
    }
    
    /* Input widgets */
    .stTextInput, .stSelectbox, .stMultiselect, .stSlider, .stDateInput {
        background-color: #1a1c23;
    }
    
    /* Plotly chart background */
    .js-plotly-plot .plotly, .js-plotly-plot .plotly div {
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================
# DATA SETUP
# ==============================================
# Corrected NSE stocks list
NSE_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", "LT.NS", "SBIN.NS",
    "BAJFINANCE.NS", "HCLTECH.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS",
    "SUNPHARMA.NS", "NTPC.NS", "ONGC.NS", "ITC.NS", "NESTLEIND.NS",
    "ADANIENT.NS", "POWERGRID.NS", "ULTRACEMCO.NS", "M&M.NS", "TATASTEEL.NS"
]

# Initialize session state
if 'selected_tickers' not in st.session_state:
    st.session_state.selected_tickers = NSE_STOCKS[:5]  # Default first 5 stocks

# ==============================================
# SIDEBAR CONTROLS
# ==============================================
with st.sidebar:
    st.image("https://seeklogo.com/images/N/national-stock-exchange-of-india-ltd-logo-0B3F1F2B0D-seeklogo.com.png", 
             width=100, use_container_width=True)
    st.title("NSE Pulse Pro")
    st.markdown("---")
    
    with st.expander("🔍 Stock Selection", expanded=True):
        selected_tickers = st.multiselect(
            "Select Stocks (Max 10)", 
            options=NSE_STOCKS,
            default=st.session_state.selected_tickers,
            max_selections=10,
            format_func=lambda x: x.replace('.NS', '')  # Clean display
        )
    
    with st.expander("📅 Date Range", expanded=True):
        end_date = dt.date.today()
        start_date = end_date - timedelta(days=180)
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input('Start Date', value=start_date)
        with col2:
            end_date = st.date_input('End Date', value=end_date)
    
    with st.expander("📊 Technical Settings", expanded=True):
        st.subheader("RSI Parameters")
        rsi_window = st.slider('RSI Period', 5, 21, 14, help="Number of periods to calculate RSI")
        col1, col2 = st.columns(2)
        with col1:
            rsi_overbought = st.slider('Overbought Level', 60, 90, 70)
        with col2:
            rsi_oversold = st.slider('Oversold Level', 10, 40, 30)
    
    st.markdown("---")
    st.caption("ℹ️ Analysis runs automatically when stocks are selected")

# ==============================================
# TECHNICAL INDICATORS
# ==============================================
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ==============================================
# MAIN ANALYSIS FUNCTION
# ==============================================
def run_analysis():
    if not selected_tickers:
        st.warning("Please select at least one stock")
        return
    
    with st.spinner('🔄 Analyzing market data...'):
        try:
            # Download data with auto_adjust explicitly set
            data = yf.download(
                selected_tickers,
                start=start_date,
                end=end_date,
                group_by='ticker',
                progress=False,
                auto_adjust=True
            )
            
            if len(selected_tickers) == 1:
                data = {selected_tickers[0]: data}
            
            results = []
            rsi_history = []
            price_history = []
            
            for ticker in selected_tickers:
                try:
                    df = data[ticker][['Close','Volume']].dropna()
                    if len(df) < 50:
                        continue
                    
                    # Calculate metrics
                    current_price = df['Close'].iloc[-1]
                    prev_close = df['Close'].iloc[-2] if len(df) > 1 else current_price
                    price_change = (current_price - prev_close) / prev_close * 100
                    overall_change = (current_price - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100
                    rsi = calculate_rsi(df['Close'], rsi_window).iloc[-1]
                    
                    results.append({
                        'Ticker': ticker.replace('.NS',''),
                        'Price': current_price,
                        'Daily Change (%)': price_change,
                        'Overall Change (%)': overall_change,
                        'RSI': rsi,
                        'Status': 'Overbought' if rsi > rsi_overbought else 
                                 'Oversold' if rsi < rsi_oversold else 'Neutral',
                        'Volume': df['Volume'].iloc[-1]
                    })
                    
                    # Store history for charts
                    rsi_series = calculate_rsi(df['Close'], rsi_window).iloc[-30:]
                    price_series = df['Close'].iloc[-30:]
                    
                    rsi_history.append({
                        'Ticker': ticker.replace('.NS',''),
                        'RSI': rsi_series,
                        'Date': rsi_series.index
                    })
                    
                    price_history.append({
                        'Ticker': ticker.replace('.NS',''),
                        'Price': price_series,
                        'Date': price_series.index
                    })
                    
                except Exception as e:
                    st.warning(f"⚠️ Skipped {ticker}: {str(e)}")
            
            if not results:
                st.error("❌ No valid results to display")
                return
                
            results_df = pd.DataFrame(results).set_index('Ticker')
            
            # ==============================================
            # DASHBOARD LAYOUT
            # ==============================================
            
            # Header with date info
            st.subheader(f"📈 Market Pulse - {end_date.strftime('%d %b %Y')}")
            
            # Row 1: Key Metrics
            cols = st.columns(4)
            metrics = [
                ("Avg Daily Change", f"{results_df['Daily Change (%)'].mean():.2f}%"),
                ("Overbought Stocks", len(results_df[results_df['Status'] == 'Overbought'])),
                ("Oversold Stocks", len(results_df[results_df['Status'] == 'Oversold'])),
                ("Top Gainer", f"{results_df['Daily Change (%)'].idxmax()} ({results_df['Daily Change (%)'].max():.2f}%)")
            ]
            
            for col, (label, value) in zip(cols, metrics):
                with col:
                    st.metric(label, value)
            
            st.markdown("---")
            
            # Row 2: Performance Overview
            st.subheader("Performance Overview")
            tab1, tab2, tab3 = st.tabs(["Top Performers", "RSI Analysis", "Detailed View"])
            
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Top Gainers
                    top_gainers = results_df.sort_values('Daily Change (%)', ascending=False).head(5)
                    fig1 = px.bar(
                        top_gainers.reset_index(),
                        x='Ticker',
                        y='Daily Change (%)',
                        color='Daily Change (%)',
                        color_continuous_scale='greens',
                        title='<b>Top 5 Gainers</b>',
                        text_auto='.2f',
                        height=350
                    )
                    fig1.update_traces(
                        textfont_size=12, 
                        textangle=0, 
                        textposition="outside",
                        marker_line_color='rgb(8,48,107)',
                        marker_line_width=1
                    )
                    fig1.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        yaxis_title="Daily Change (%)",
                        xaxis_title="",
                        margin=dict(t=40, b=20, l=20, r=20)
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    # Top Losers
                    top_losers = results_df.sort_values('Daily Change (%)').head(5)
                    fig2 = px.bar(
                        top_losers.reset_index(),
                        x='Ticker',
                        y='Daily Change (%)',
                        color='Daily Change (%)',
                        color_continuous_scale='reds',
                        title='<b>Top 5 Losers</b>',
                        text_auto='.2f',
                        height=350
                    )
                    fig2.update_traces(
                        textfont_size=12, 
                        textangle=0, 
                        textposition="outside",
                        marker_line_color='rgb(8,48,107)',
                        marker_line_width=1
                    )
                    fig2.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        yaxis_title="Daily Change (%)",
                        xaxis_title="",
                        margin=dict(t=40, b=20, l=20, r=20)
                    )
                    st.plotly_chart(fig2, use_container_width=True)
            
            with tab2:
                col1, col2 = st.columns([0.6, 0.4])
                
                with col1:
                    # RSI Distribution
                    fig3 = px.histogram(
                        results_df.reset_index(),
                        x='RSI',
                        nbins=20,
                        title='<b>RSI Value Distribution</b>',
                        color_discrete_sequence=['#636EFA'],
                        height=400
                    )
                    fig3.add_vline(
                        x=rsi_overbought, 
                        line_dash="dash", 
                        line_color="red", 
                        annotation_text="Overbought",
                        annotation_position="top"
                    )
                    fig3.add_vline(
                        x=rsi_oversold, 
                        line_dash="dash", 
                        line_color="green", 
                        annotation_text="Oversold",
                        annotation_position="top"
                    )
                    fig3.add_vline(x=50, line_dash="dot", line_color="gray")
                    fig3.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        yaxis_title="Number of Stocks",
                        xaxis_title="RSI Value",
                        margin=dict(t=40, b=20, l=20, r=20)
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                
                with col2:
                    # RSI Status Pie Chart
                    status_counts = results_df['Status'].value_counts()
                    fig4 = px.pie(
                        status_counts,
                        values=status_counts.values,
                        names=status_counts.index,
                        title='<b>RSI Status Distribution</b>',
                        color=status_counts.index,
                        color_discrete_map={
                            'Overbought':'#EF553B', 
                            'Oversold':'#00CC96', 
                            'Neutral':'#636EFA'
                        },
                        height=400
                    )
                    fig4.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        marker=dict(line=dict(color='#ffffff', width=1)),
                        hole=0.4,
                        textfont=dict(color='white')
                    )
                    fig4.update_layout(
                        showlegend=False,
                        margin=dict(t=40, b=20, l=20, r=20),
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    st.plotly_chart(fig4, use_container_width=True)
            
            with tab3:
                # Detailed View with Filters
                st.subheader("Stock Details")
                
                # Filters
                col1, col2 = st.columns(2)
                with col1:
                    ticker_filter = st.multiselect(
                        "Filter by Ticker",
                        options=results_df.index.unique(),
                        default=results_df.index.tolist(),
                        format_func=lambda x: x
                    )
                
                with col2:
                    status_filter = st.multiselect(
                        "Filter by RSI Status",
                        options=['Overbought', 'Neutral', 'Oversold'],
                        default=['Overbought', 'Neutral', 'Oversold']
                    )
                
                # Apply filters
                filtered_df = results_df[
                    (results_df.index.isin(ticker_filter)) &
                    (results_df['Status'].isin(status_filter))
                ]
                
                # Format and style the dataframe
                def color_status(val):
                    color = 'red' if val == 'Overbought' else 'green' if val == 'Oversold' else 'blue'
                    return f'color: {color}; font-weight: bold'
                
                def color_negative(val):
                    return 'color: #ff4b4b' if isinstance(val, (int, float)) and val < 0 else 'color: #4bb543'
                
                styled_df = filtered_df.style \
                    .format({
                        'Price': '₹{:.2f}',
                        'Daily Change (%)': '{:+.2f}%',
                        'Overall Change (%)': '{:+.2f}%',
                        'RSI': '{:.1f}',
                        'Volume': '{:,}'
                    }) \
                    .map(color_status, subset=['Status']) \
                    .map(color_negative, subset=['Daily Change (%)', 'Overall Change (%)'])
                
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    height=500,
                    column_order=['Price', 'Daily Change (%)', 'Overall Change (%)', 'RSI', 'Status', 'Volume']
                )
            
            # Row 3: Historical Trends
            st.markdown("---")
            st.subheader("Historical Trends")
            
            if len(rsi_history) > 0:
                selected_ticker = st.selectbox(
                    "Select stock to view historical trends",
                    options=[x['Ticker'] for x in rsi_history],
                    index=0
                )
                
                # Find the selected stock's data
                selected_rsi = next((x for x in rsi_history if x['Ticker'] == selected_ticker), None)
                selected_price = next((x for x in price_history if x['Ticker'] == selected_ticker), None)
                
                if selected_rsi and selected_price:
                    # Create figure with secondary y-axis
                    fig5 = go.Figure()
                    
                    # Add price trace
                    fig5.add_trace(go.Scatter(
                        x=selected_price['Date'],
                        y=selected_price['Price'],
                        name='Price (₹)',
                        line=dict(color='#4e79a7', width=2),
                        yaxis='y1'
                    ))
                    
                    # Add RSI trace
                    fig5.add_trace(go.Scatter(
                        x=selected_rsi['Date'],
                        y=selected_rsi['RSI'],
                        name='RSI',
                        line=dict(color='#f28e2b', width=2),
                        yaxis='y2'
                    ))
                    
                    # Add RSI reference lines
                    fig5.add_hline(
                        y=rsi_overbought,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Overbought",
                        annotation_position="top right",
                        yref="y2"
                    )
                    
                    fig5.add_hline(
                        y=rsi_oversold,
                        line_dash="dash",
                        line_color="green",
                        annotation_text="Oversold",
                        annotation_position="bottom right",
                        yref="y2"
                    )
                    
                    fig5.add_hline(
                        y=50,
                        line_dash="dot",
                        line_color="gray",
                        yref="y2"
                    )
                    
                    # Update layout
                    fig5.update_layout(
                        title=f'<b>{selected_ticker} - Price & RSI Trend (Last 30 Days)</b>',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        hovermode='x unified',
                        yaxis=dict(
                            title='Price (₹)',
                            side='left',
                            showgrid=True,
                            gridcolor='rgba(255,255,255,0.1)'
                        ),
                        yaxis2=dict(
                            title='RSI',
                            side='right',
                            overlaying='y',
                            range=[0, 100],
                            showgrid=False
                        ),
                        height=400,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        margin=dict(t=60, b=20, l=20, r=40)
                    )
                    
                    st.plotly_chart(fig5, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")

# ==============================================
# RUN THE APPLICATION
# ==============================================
if len(selected_tickers) > 0:
    run_analysis()
else:
    st.info("ℹ️ Please select stocks from the sidebar to begin analysis")

# Footer
st.markdown("---")
st.caption(f"""
    **NSE Pulse Pro** | Data from Yahoo Finance | 
    Last Updated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} | 
    [GitHub Repo](#)
""")