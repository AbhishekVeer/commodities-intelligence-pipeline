import streamlit as st
import yfinance as yf
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import concurrent.futures
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# -----------------------------------------
# 1. PAGE CONFIG & SECRETS
# -----------------------------------------
st.set_page_config(page_title="Commodities Intelligence Pipeline V2", layout="wide")
st.title("🛢️ AI-Driven Commodities Intelligence Pipeline (V2)")

try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    news_api_key = st.secrets["NEWS_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("Missing API keys. Please add OPENAI_API_KEY and NEWS_API_KEY to your Streamlit secrets.")
    st.stop()

# -----------------------------------------
# 2. V2 PYDANTIC SCHEMA & LLM SETUP
# -----------------------------------------
class MarketSignalV2(BaseModel):
    target_commodity: str = Field(description="Strictly identify the commodity. If not explicitly Crude Oil or WTI, output 'Other'.")
    fundamental_category: str = Field(description="Categorize: Supply Disruption, Demand Shift, Regulatory, Macro")
    sentiment: str = Field(description="Bullish, Bearish, or Neutral")
    severity_score: int = Field(description="Impact severity from 1 to 10. (e.g., geopolitical war = 9, minor inventory draw = 3)")
    reasoning: str = Field(description="One sentence justification for the score.")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)
structured_llm_v2 = llm.with_structured_output(MarketSignalV2)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert energy commodities quantitative analyst. Analyze the provided news summary and extract the market fundamentals."),
    ("human", "Headline: {title}\nSummary: {summary}")
])
extraction_chain_v2 = prompt | structured_llm_v2

# -----------------------------------------
# 3. DATA INGESTION ENGINE
# -----------------------------------------
@st.cache_data(ttl=3600)
def fetch_market_data(ticker="CL=F", days=45):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    df = yf.download(ticker, start=start_date, end=end_date)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    return df[['Date', 'Close']]

@st.cache_data(ttl=3600)
def scrape_energy_news(days=30):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    url = (f"https://newsapi.org/v2/everything?q=crude oil OR energy markets&"
           f"from={start_date.strftime('%Y-%m-%d')}&to={end_date.strftime('%Y-%m-%d')}&"
           f"sortBy=relevancy&language=en&apiKey={news_api_key}")
    
    response = requests.get(url)
    if response.status_code != 200:
        return pd.DataFrame()
        
    articles = response.json().get('articles', [])
    news_items = []
    
    for article in articles[:20]:
        summary = article.get('description') or article.get('title')
        pub_date = pd.to_datetime(article['publishedAt']).date()
        news_items.append({"Date": pub_date, "Headline": article['title'], "Summary": summary})
    return pd.DataFrame(news_items)

# -----------------------------------------
# 4. MULTITHREADED LLM PROCESSING ENGINE
# -----------------------------------------
@st.cache_data(ttl=3600)
def process_unstructured_news_v2(news_dataframe):
    def parse_row(row):
        try:
            result = extraction_chain_v2.invoke({
                "title": row['Headline'], 
                "summary": row['Summary']
            })
            return {
                "Date": row['Date'],
                "Headline": row['Headline'],
                "Commodity": result.target_commodity,
                "Category": result.fundamental_category,
                "Sentiment": result.sentiment,
                "Severity": result.severity_score,
                "Reasoning": result.reasoning
            }
        except Exception:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        rows = [row for _, row in news_dataframe.iterrows()]
        results = list(executor.map(parse_row, rows))
        
    valid_results = [res for res in results if res is not None]
    return pd.DataFrame(valid_results)

# -----------------------------------------
# 5. V2 EVALUATION METRIC (T+3 Horizon)
# -----------------------------------------
def evaluate_signals_v2(price_data, signal_data):
    df_price = price_data.sort_values('Date').copy()
    
    df_price['T1_Close'] = df_price['Close'].shift(-1)
    df_price['T3_Close'] = df_price['Close'].shift(-3)
    
    eval_df = pd.merge(signal_data, df_price, on='Date', how='inner')
    
    eval_df = eval_df[
        (eval_df['Commodity'].isin(['Crude Oil', 'WTI'])) & 
        (eval_df['Sentiment'].isin(['Bullish', 'Bearish'])) &
        (eval_df['Severity'] >= 6) 
    ].dropna(subset=['T3_Close'])
    
    if len(eval_df) == 0:
        return eval_df, 0, 0, 0.0

    def check_t3_accuracy(row):
        direction_t3 = 'Up' if row['T3_Close'] > row['Close'] else 'Down'
        if row['Sentiment'] == 'Bullish' and direction_t3 == 'Up':
            return 1
        elif row['Sentiment'] == 'Bearish' and direction_t3 == 'Down':
            return 1
        return 0
            
    eval_df['T3_Is_Correct'] = eval_df.apply(check_t3_accuracy, axis=1)
    
    total_signals = len(eval_df)
    correct_t3 = eval_df['T3_Is_Correct'].sum()
    hit_rate_t3 = (correct_t3 / total_signals) * 100 if total_signals > 0 else 0.0
    
    return eval_df, total_signals, correct_t3, hit_rate_t3

# -----------------------------------------
# 6. APP EXECUTION & UI RENDER
# -----------------------------------------
with st.spinner("Fetching market data and scraping news..."):
    price_df = fetch_market_data()
    news_df = scrape_energy_news()

if not news_df.empty:
    with st.spinner("Multithreaded LLM Agents extracting market signals..."):
        structured_news_df = process_unstructured_news_v2(news_df)
    
    eval_df, total, correct, hit_rate = evaluate_signals_v2(price_df, structured_news_df)
    
    st.subheader("Filtered Pipeline Performance (T+3 Backtest | Severity ≥ 6)")
    col1, col2, col3 = st.columns(3)
    col1.metric("High-Conviction Signals", total)
    col2.metric("Correct Predictions (T+3)", correct)
    col3.metric("Directional Hit Rate", f"{hit_rate:.2f}%")
    
    st.subheader("Price Action vs. High-Conviction Market Fundamentals")
    dashboard_df = pd.merge(price_df, structured_news_df, on='Date', how='left')
    
    dashboard_df = dashboard_df[
        (dashboard_df['Severity'] >= 6) | (dashboard_df['Severity'].isna())
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dashboard_df['Date'], y=dashboard_df['Close'],
        mode='lines+markers', name='WTI Crude (USD)',
        line=dict(color='#2962FF', width=2)
    ))
    
    bullish = dashboard_df[dashboard_df['Sentiment'] == 'Bullish']
    fig.add_trace(go.Scatter(
        x=bullish['Date'], y=bullish['Close'], mode='markers', name='Bullish Signal (High Conviction)',
        marker=dict(color='#00C853', size=14, symbol='triangle-up'),
        text="Sev " + bullish['Severity'].astype(str) + ": " + bullish['Headline'] + "<br>" + bullish['Reasoning'], 
        hoverinfo='text'
    ))
    
    bearish = dashboard_df[dashboard_df['Sentiment'] == 'Bearish']
    fig.add_trace(go.Scatter(
        x=bearish['Date'], y=bearish['Close'], mode='markers', name='Bearish Signal (High Conviction)',
        marker=dict(color='#D50000', size=14, symbol='triangle-down'),
        text="Sev " + bearish['Severity'].astype(str) + ": " + bearish['Headline'] + "<br>" + bearish['Reasoning'], 
        hoverinfo='text'
    ))
    
    fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)", template="plotly_dark", height=500)
    st.plotly_chart(fig, width='stretch')
    
    st.subheader("Structured V2 AI Outputs")
    st.dataframe(structured_news_df, width='stretch')
    
    st.subheader("Evaluation Ledger")
    if not eval_df.empty:
        display_cols = ['Date', 'Headline', 'Sentiment', 'Severity', 'Close', 'T3_Close', 'T3_Is_Correct']
        st.dataframe(eval_df[display_cols], width='stretch')
else:
    st.warning("No news data was returned by the API.")
