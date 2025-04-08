import streamlit as st

# Import libraries
import numpy as np 
import pandas as pd 
import seaborn as sns

import statsmodels.api as sm
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tools.eval_measures import rmse, aic

import plotly.express as px 
import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime  
from dateutil.relativedelta import relativedelta  

@st.cache_data
def get_dd():
    fp = 'hrc_price_CN_JP.csv'
    return pd.read_csv(fp)

def create_upside_downside(df, column_name, upside_pct, downside_pct):
    df["upside"] = df[column_name] * (1 + upside_pct / 100)
    df["downside"] = df[column_name] * (1 - downside_pct / 100)
    return df

st.set_page_config(    
    page_title="HRC Price Predict Model Dashboard",
    page_icon="⭕",
    layout="wide"
)

st.markdown(f'''
    <h1 style="font-size: 25px; color: white; text-align: center; background: #0E539A; border-radius: .5rem; margin-bottom: 1rem;">
    HRC Price Predict Model Dashboard
    </h1>''', unsafe_allow_html=True)

col = st.columns([1.2, 3])
col1 = col[1].columns(3)
col0 = col[0].columns(2)

# User inputs
sea_freight = col0[0].number_input("**Sea Freight**",value=10)
exchange_rate = col0[1].number_input("**Exchange Rate (Rs/USD)**",value=0.1)
upside_factor = col0[0].number_input("**Upside (%)**", value=5)
downside_factor = col0[1].number_input("**Downside (%)**", value=5)

df = get_dd()
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(by="Date")

final_df = df.dropna()
final_df.set_index('Date', inplace=True)

maxlags = col1[0].number_input("**Maxlags**", value=12, min_value=1, max_value=100)
months = col1[1].number_input(f"**Months ahead (Started in {final_df.index.tolist()[-1].strftime('%Y-%m-%d')})**", value=17, min_value=1, max_value=50)
country = col1[2].multiselect("Please choose country", ["China", "Japan"], ["China", "Japan"])

final_df_differenced = final_df.diff().dropna()
model = VAR(final_df_differenced)
model_fitted = model.fit(maxlags)
lag_order = model_fitted.k_ar
forecast_input = final_df_differenced.values[-lag_order:]

# Forecast
df_forecast = pd.DataFrame(
    model_fitted.forecast(y=forecast_input, steps=months),
    index=pd.date_range(start=final_df.index[-1], periods=months, freq='MS'),
    columns=final_df.columns + '_1d'
)

def invert_transformation(df_train, df_forecast, second_diff=False):
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:        
        if second_diff:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc

df_forecast_processed = invert_transformation(final_df, df_forecast)

# Extract result forecast
df_res = df_forecast_processed[["China HRC (FOB, $/t)_forecast", "Japan HRC (FOB, $/t)_forecast"]].reset_index()
last_row = df_res.iloc[-1]
month = last_row[0]
china_hrc = last_row[1]
japan_hrc = last_row[2]

# Upside/downside added
df_res = create_upside_downside(df_res, "China HRC (FOB, $/t)_forecast", upside_factor, downside_factor)

# Build main plot
fig = go.Figure()

if "China" in country:
    fig.add_trace(go.Scatter(x=df_res["Date"], y=df_res["China HRC (FOB, $/t)_forecast"],
                             mode='lines', name='China Forecast', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df_res["Date"], y=df_res["upside"],
                             mode='lines', name='China Upside', line=dict(dash='dot', color='green')))
    fig.add_trace(go.Scatter(x=df_res["Date"], y=df_res["downside"],
                             mode='lines', name='China Downside', line=dict(dash='dot', color='red')))

if "Japan" in country:
    fig.add_trace(go.Scatter(x=df_res["Date"], y=df_res["Japan HRC (FOB, $/t)_forecast"],
                             mode='lines', name='Japan Forecast', line=dict(color='orange')))

fig.update_layout(title="HRC Price Forecast with Upside/Downside",
                  xaxis_title="Date",
                  yaxis_title="Price ($/t)",
                  legend_title="Legend",
                  height=500)

col[1].plotly_chart(fig, use_container_width=True)

# Price table and landed price calculation
china_land_price = exchange_rate*(10+1.01*(china_hrc+sea_freight)*(1+1.1*7.5))+500
japan_land_price = exchange_rate*(10+1.01*(japan_hrc+sea_freight)*(1+1.1*0))+500

if 'China' in country:
    col[0].markdown(f'<span style="color:#0E549B;font-weight:bold;">China landed price is: {round(china_land_price)} Rs/t</span>', unsafe_allow_html=True)

if 'Japan' in country:
    col[0].markdown(f'<span style="color:#C93B3B;font-weight:bold;">Japan landed price is: {round(japan_land_price)} Rs/t</span>', unsafe_allow_html=True)
