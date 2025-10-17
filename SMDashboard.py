import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go

st.title("Stock Market Data Analysis")
st.write("Upload a CSV containing Date, Open, High, Low, Close, Volume.")

uploaded_file = st.file_uploader('Upload CSV', type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    required_cols = {"Date", "Open", "High", "Low", "Close", "Volume"}
    if not required_cols.issubset(df.columns):
        st.error(f"CSV must have columns: {', '.join(required_cols)}")
    else:
        df = df[list(required_cols)].dropna()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

        st.subheader("Cleaned Data Preview (first 10 rows)")
        st.dataframe(df.head(10))

        # Moving Average Strategy
        df['MA20'] = df['Close'].rolling(20, min_periods=1).mean()
        df['MA50'] = df['Close'].rolling(50, min_periods=1).mean()
        df['MA_signal'] = np.where(df['MA20'] > df['MA50'], 1, np.where(df['MA20'] < df['MA50'], -1, None))

        # Fibonacci Retracement Strategy
        high, low = df['High'].max(), df['Low'].min()
        fib_618 = high - 0.618 * (high - low)
        fib_236 = high - 0.236 * (high - low)
        df['Fib_signal'] = 0
        df.loc[np.isclose(df['Close'], fib_618, atol=0.02 * df['Close']), 'Fib_signal'] = 1
        df.loc[np.isclose(df['Close'], fib_236, atol=0.02 * df['Close']), 'Fib_signal'] = -1

        # Support and Resistance Strategy
        df['Support'] = df['Low'].rolling(20, min_periods=1).min()
        df['Resistance'] = df['High'].rolling(20, min_periods=1).max()
        df['SR_signal'] = 0
        df.loc[np.abs(df['Close'] - df['Support']) < 0.01 * df['Support'], 'SR_signal'] = 1
        df.loc[np.abs(df['Close'] - df['Resistance']) < 0.01 * df['Resistance'], 'SR_signal'] = -1

        # Show signals corresponding to price values
        signal_data = df[['Close', 'MA_signal', 'Fib_signal', 'SR_signal']].dropna(how='all')
        st.subheader("Signals Corresponding to Price Values")
        st.dataframe(signal_data.tail(15))

        # Moving Average Strategy Plot
        st.subheader("Moving Average Crossover Strategy")
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Candlestick(
            x=df['Date'], open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'], name='Candlestick'))
        fig_ma.add_trace(go.Scatter(x=df['Date'], y=df['MA20'], mode='lines', name='MA 20', line=dict(color='blue')))
        fig_ma.add_trace(go.Scatter(x=df['Date'], y=df['MA50'], mode='lines', name='MA 50', line=dict(color='orange')))

        buys_ma = df[df['MA_signal'] == 1]
        sells_ma = df[df['MA_signal'] == -1]
        fig_ma.add_trace(go.Scatter(
            x=buys_ma['Date'], y=buys_ma['Close'], mode='markers',
            marker_symbol='triangle-up', marker_color='limegreen', marker_size=12, name='Buy'))
        fig_ma.add_trace(go.Scatter(
            x=sells_ma['Date'], y=sells_ma['Close'], mode='markers',
            marker_symbol='triangle-down', marker_color='red', marker_size=12, name='Sell'))

        fig_ma.update_layout(xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig_ma, use_container_width=True)

        # Fibonacci Retracement Plot
        st.subheader("Fibonacci Retracement Strategy")
        fig_fib = go.Figure()
        fig_fib.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close Price', line=dict(color='white')))
        fig_fib.add_hline(y=fib_618, line_dash="dot", annotation_text='Fib 61.8%', line_color='blue')
        fig_fib.add_hline(y=fib_236, line_dash="dot", annotation_text='Fib 23.6%', line_color='purple')

        buys_fib = df[df['Fib_signal'] == 1]
        sells_fib = df[df['Fib_signal'] == -1]
        fig_fib.add_trace(go.Scatter(
            x=buys_fib['Date'], y=buys_fib['Close'], mode='markers',
            marker_symbol='star', marker_color='green', marker_size=11, name=1))
        fig_fib.add_trace(go.Scatter(
            x=sells_fib['Date'], y=sells_fib['Close'], mode='markers',
            marker_symbol='star', marker_color='red', marker_size=11, name=-1))
        fig_fib.update_layout(xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig_fib, use_container_width=True)

        # Support and Resistance Plot
        st.subheader("Support and Resistance Strategy")
        fig_sr = go.Figure()
        fig_sr.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close', line=dict(color='black')))
        fig_sr.add_trace(go.Scatter(x=df['Date'], y=df['Support'], mode='lines', name='Support', line=dict(color='green', dash='dash')))
        fig_sr.add_trace(go.Scatter(x=df['Date'], y=df['Resistance'], mode='lines', name='Resistance', line=dict(color='red', dash='dash')))

        buys_sr = df[df['SR_signal'] == 1]
        sells_sr = df[df['SR_signal'] == -1]
        fig_sr.add_trace(go.Scatter(
            x=buys_sr['Date'], y=buys_sr['Close'], mode='markers',
            marker_symbol='circle', marker_color='green', marker_size=11, name=1))
        fig_sr.add_trace(go.Scatter(
            x=sells_sr['Date'], y=sells_sr['Close'], mode='markers',
            marker_symbol='x', marker_color='red', marker_size=11, name=-1))
        fig_sr.update_layout(xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig_sr, use_container_width=True)

        # Download signals CSV
        csv = signal_data.to_csv(index=False)
        st.download_button("Download CSV with Price Corresponding Signals", csv, "signals_by_price.csv", "text/csv")

import streamlit as st
import yfinance as yf

def get_ticker(company_name):
    # Simple manual mapping for demo (extend or use search API in real app)
    companies = {
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Google": "GOOGL",
        "Amazon": "AMZN",
        "Tesla": "TSLA"
    }
    return companies.get(company_name, "")

def get_stock_summary(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    summary = f"""
    **{info['longName']} ({ticker})**

    - Current Price: ${info['currentPrice']}
    - Market Cap: {info['marketCap']:,}
    - PE Ratio: {info.get('trailingPE', 'N/A')}
    - Dividend Yield: {info.get('dividendYield', 'N/A')}
    - 52 Week High: ${info.get('fiftyTwoWeekHigh', 'N/A')}
    - 52 Week Low: ${info.get('fiftyTwoWeekLow', 'N/A')}
    """
    return summary

def main():
    st.title("Simple Stock Summary AI")
    company_name = st.text_input("Enter Company Name (e.g. Apple, Microsoft)")
    if company_name:
        ticker = get_ticker(company_name)
        if ticker:
            summary = get_stock_summary(ticker)
            st.markdown(summary)
        else:
            st.error("Company not found. Please try Apple, Microsoft, Google, Amazon, or Tesla.")

if __name__ == "__main__":
    main()
