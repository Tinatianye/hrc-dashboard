import streamlit as st
import numpy as np 
import pandas as pd 
import plotly.graph_objects as go
import warnings
from statsmodels.tsa.api import VAR
from datetime import datetime  
from dateutil.relativedelta import relativedelta  

warnings.filterwarnings("ignore")

st.set_page_config(    
    page_title="HRC Price Predict Model Dashboard",
    page_icon="⭕",
    layout="wide"
)

@st.cache_data
def get_dd():
    return pd.read_csv('hrc_price_CN_JP.csv')

st.markdown(f'''
    <h1 style="font-size: 25px; color: white; text-align: center; background: #0E539A; border-radius: .5rem; margin-bottom: 1rem;">
    HRC Price Predict Model Dashboard
    </h1>''', unsafe_allow_html=True)

# === Layout setup ===
left_col, right_col = st.columns([1, 3])

with left_col:
    st.subheader("Model Parameters")
    sea_freight = st.number_input("Sea Freight", value=10)
    exchange_rate = st.number_input("Exchange Rate (Rs/USD)", value=0.1)
    upside_pct = st.number_input("Upside (%)", value=10)
    downside_pct = st.number_input("Downside (%)", value=10)
    maxlags = st.number_input("Maxlags", value=12, min_value=1, max_value=100)

    df = get_dd()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by="Date")
    final_df = df.dropna()
    final_df.set_index('Date', inplace=True)

    months = st.number_input(
        f"Months ahead (Started in {final_df.index[-1].strftime('%Y-%m-%d')})",
        value=17, min_value=1, max_value=50)
    country = st.multiselect("Please choose country", ["China", "Japan"], ["China", "Japan"])
