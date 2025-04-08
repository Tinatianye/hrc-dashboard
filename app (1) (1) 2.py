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

import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime  
from dateutil.relativedelta import relativedelta  

@st.cache_data
def get_dd():
    fp = 'hrc_price_CN_JP.csv'
    return pd.read_csv(fp)

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

df = get_dd()

sea_freight = col0[0].number_input("**Sea Freight**",value=10)
exchange_rate = col0[1].number_input("**Exchange Rate (Rs/USD)**",value=0.1)

# ✅ 新增：Upside / Downside 百分比输入
upside_pct = col0[0].number_input("**Upside (%)**", value=10)
downside_pct = col0[1].number_input("**Downside (%)**", value=10)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(by="Date")
final_df = df.dropna()
final_df.set_index('Date', inplace=True)

maxlags = col1[0].number_input("**Maxlags**", value=12, min_value=1, max_value=100)
months = col1[1].number_input(f"**Months ahead (Started in {final_df.index.tolist()[-1].strftime('%Y-%m-%d')})**", value=17, min_value=1, max_value=50)
country = col1[2].multiselect("Please choose country", ["China", "Japan"], ["China", "Japan"])

final_df_differenced = final_df.diff().dropna()

model = VAR(final_df_differenced)
x = model.select_order(maxlags=maxlags)
model_fitted = model.fit(maxlags)

lag_order = model_fitted.k_ar
forecast_input = final_df_differenced.values[-lag_order:]


def invert_transformation(df_train, df_forecast):
    df_fc = pd.DataFrame()
    for i, col in enumerate(df_train.columns):
        last_value = df_train[col].iloc[-1]
        df_fc[f"{col}_forecast"] = last_value + df_forecast.iloc[:, i].cumsum()
    return df_fc


fc = model_fitted.forecast(y=forecast_input, steps=months)
fc_period = pd.date_range(
    start=final_df.index.tolist()[-1],
    end=final_df.index.tolist()[-1]+relativedelta(months=months-1), freq='MS')
df_forecast = pd.DataFrame(fc, index=fc_period, columns=final_df.columns + '_1d')
df_forecast_processed = invert_transformation(final_df, df_forecast)

df_res = df_forecast_processed[["China HRC (FOB, $/t)_forecast", "Japan HRC (FOB, $/t)_forecast"]]
df_res = df_res.reset_index()
df_res.rename(columns={"index": "Date"}, inplace=True)

# ✅ 新增：上下行预测区间计算函数
def apply_upside_downside(df, column, up_pct, down_pct):
    df[f'{column}_upside'] = df[column] * (1 + up_pct / 100)
    df[f'{column}_downside'] = df[column] * (1 - down_pct / 100)
    return df

if "China" in country:
    df_res = apply_upside_downside(df_res, "China HRC (FOB, $/t)_forecast", upside_pct, downside_pct)

if "Japan" in country:
    df_res = apply_upside_downside(df_res, "Japan HRC (FOB, $/t)_forecast", upside_pct, downside_pct)

# ✅ 使用 Plotly 展示上下区间为填充带状区域
fig = go.Figure()

if "China" in country:
    fig.add_trace(go.Scatter(
        x=df_res["Date"],
        y=df_res["China HRC (FOB, $/t)_forecast"],
        name="China Forecast",
        line=dict(color="#0E549B")
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([df_res["Date"], df_res["Date"][::-1]]),
        y=pd.concat([df_res["China HRC (FOB, $/t)_upside"], df_res["China HRC (FOB, $/t)_downside"][::-1]]),
        fill='toself',
        fillcolor='rgba(14, 84, 155, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name="China Upside/Downside"
    ))

if "Japan" in country:
    fig.add_trace(go.Scatter(
        x=df_res["Date"],
        y=df_res["Japan HRC (FOB, $/t)_forecast"],
        name="Japan Forecast",
        line=dict(color="red")
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([df_res["Date"], df_res["Date"][::-1]]),
        y=pd.concat([df_res["Japan HRC (FOB, $/t)_upside"], df_res["Japan HRC (FOB, $/t)_downside"][::-1]]),
        fill='toself',
        fillcolor='rgba(255,0,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name="Japan Upside/Downside"
    ))

fig.update_layout(
    title={
        'text': "/".join(country) + " Forecasting HRC Prices (with Upside/Downside)",
        'x': 0.5,
        'y': 0.96,
        'xanchor': 'center'
    },
    height=500,
    margin=dict(t=30),
    legend=dict(
        title="",
        yanchor="top",
        y=0.99,
        xanchor="center",
        x=0.5,
        orientation="h"
    ),
    xaxis={"title": "Date"},
    paper_bgcolor='rgba(0, 0, 0, 0)',
    plot_bgcolor='rgba(0, 0, 0, 0)'
)

col[1].plotly_chart(fig, use_container_width=True)


last_row = df_res.iloc[-1]
china_hrc = last_row["China HRC (FOB, $/t)_forecast"]
japan_hrc = last_row["Japan HRC (FOB, $/t)_forecast"]

china_columns = ['HRC FOB China', 'Sea Freight', 'Basic Customs Duty (%)', 'LC charges & Port Charges',
                 'Exchange Rate (Rs/USD)', 'Freight from port to city']
china_data = [china_hrc, sea_freight, 7.5, 10, exchange_rate, 500]

japan_columns = ['HRC FOB Japan', 'Sea Freight', 'Basic Customs Duty (%)', 'LC charges & Port Charges',
                 'Exchange Rate (Rs/USD)', 'Freight from port to city']
japan_data = [japan_hrc, sea_freight, 0, 10, exchange_rate, 500]

china_df = pd.DataFrame(china_data, index=china_columns).reset_index()
china_df.columns = ['Factors', 'Value']

japan_df = pd.DataFrame(japan_data, index=japan_columns).reset_index()
japan_df.columns = ['Factors', 'Value']

china_land_price = exchange_rate*(10+1.01*(china_hrc+sea_freight)*(1+1.1*7.5))+500
japan_land_price = exchange_rate*(10+1.01*(japan_hrc+sea_freight)*(1+1.1*0))+500

if 'China' in country:
    col[0].write("**China**")
    col[0].dataframe(china_df, hide_index=True)
    col[0].markdown(f'<span style="color:#0E549B;font-weight:bold;">China landed price is: {round(china_land_price)} Rs/t</span>', unsafe_allow_html=True)

if 'Japan' in country:
    col[0].write("**Japan**")
    col[0].dataframe(japan_df, hide_index=True)
    col[0].markdown(f'<span style="color:#C93B3B;font-weight:bold;">Japan landed price is: {round(japan_land_price)} Rs/t</span>', unsafe_allow_html=True)
