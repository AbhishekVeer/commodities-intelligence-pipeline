# Commodities-intelligence-pipeline
An automated, AI-driven commodities intelligence pipeline that scrapes unstructured market news, extracts structured fundamental trading signals using LLMs, and evaluates their predictive power against crude oil (WTI) price action.

# 🛢️ AI-Driven Commodities Intelligence Pipeline

**Live Application:** https://commodities-intelligence-pipeline-ozbvpuxwnuewnjro3gevqb.streamlit.app/

## 📌 Project Overview
This project is an automated agentic workflow designed to bridge the gap between unstructured macroeconomic news and quantitative commodities trading. It systematically scrapes daily market headlines, leverages Large Language Models (LLMs) to extract structured fundamental signals, and backtests those signals against historical WTI crude oil price action. 

Designed as a V2 architecture, this pipeline moves beyond simple zero-shot extraction by introducing multithreaded parallel processing, strict target verification, and multi-horizon (T+3) forward evaluation filtering for high-conviction market shocks.

## 🏗️ System Architecture

* **Data Ingestion Engine:** Automated Python scraper utilizing `NewsAPI` to aggregate real-time unstructured macro and energy-specific headlines, alongside `yfinance` for historical WTI (CL=F) price action.
* **Multithreaded Agentic Structuring:** A `LangChain` orchestration pipeline utilizing `gpt-4o-mini`. It uses `ThreadPoolExecutor` for parallel processing and strict `Pydantic` schemas to categorize market fundamentals (Supply, Demand, Regulatory), assign a quantitative Severity Score (1-10), and extract qualitative sentiment.
* **Evaluation Loop:** Programmatic `Pandas` backtesting script that evaluates the LLM's predictive power. The V2 logic calculates rolling T+3 forward returns and strictly filters for high-severity (≥ 6) signals directly targeting Crude Oil to reduce market noise.
* **Visualization:** Interactive `Plotly` and `Streamlit` dashboard overlaying extracted, high-conviction news sentiment directly onto the commodity price chart for visual alpha confirmation.

## 📊 V2 Evaluation Metrics & Logic
While a V1 baseline evaluation (T+1, unweighted) yielded a 45.45% hit rate due to broad macroeconomic noise, the V2 pipeline introduces quantitative rigidity:

1. **Noise Filtering:** The LLM actively drops articles not explicitly tied to WTI or Crude Oil fundamentals.
2. **Conviction Thresholds:** Only signals with a severity score of 6 or higher are evaluated.
3. **Lagged Pricing (T+3):** Acknowledging that systemic macro news takes time to price into the physical commodity, accuracy is measured against the T+3 closing price.

## 🚀 Local Installation & Execution

### If you wish to run this pipeline locally rather than utilizing the cloud deployment:

1. **Clone the repository**:
   ```bash  
   git clone [https://github.com/your-username/commodities-intelligence-pipeline.git](https://github.com/your-username/commodities-intelligence-pipeline.git)
   cd commodities-intelligence-pipeline

**Install dependencies:**

1. **Bash**
pip install -r requirements.txt

3. **Set your API Keys:**
Create a .streamlit/secrets.toml file in the root directory and add:

Ini, TOML
OPENAI_API_KEY = "your-openai-key"
NEWS_API_KEY = "your-newsapi-key"

3. **Launch the application:**
Bash
streamlit run app.py

## 🛠️ Technology Stack
Language: Python 3.10+
AI/ML: LangChain, OpenAI (gpt-4o-mini), Pydantic

Data & Analytics: Pandas, NumPy, yfinance

Deployment & UI: Streamlit Community Cloud, Plotly



