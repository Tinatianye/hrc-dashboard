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

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime  
from dateutil.relativedelta import relativedelta  

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

uploaded_file = col[0].file_uploader("Choose a file", type=["csv"])

file_path = 'wo_na.csv'
dd = pd.read_csv(file_path)

ff = False
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if str(df.columns.tolist())==str(dd.columns.tolist()):
        ff = True
    else:
        file_path = 'wo_na.csv'
        df = pd.read_csv(file_path)
        col[0].warning("Columns error, please keep the headers consistent!")
else:
    file_path = 'wo_na.csv'
    df = pd.read_csv(file_path)
    col[0].warning("No file be upload!")



# Forecast logic
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(by="Date")

final_df = df.copy()
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

fc = model_fitted.forecast(y=forecast_input, steps=months)
fc_period = pd.date_range(start=final_df.index.tolist()[-1]+relativedelta(months=1), 
                          end=final_df.index.tolist()[-1]+relativedelta(months=months), freq='MS')
df_forecast = pd.DataFrame(fc, index=fc_period, columns=final_df.columns + '_1d')

def invert_transformation(df_train, df_forecast, second_diff=False):
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:        
        if second_diff:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc

df_forecast_processed = invert_transformation(final_df, df_forecast)

# Plotting

def fun(x):
    d1 = final_df[[f'{x} HRC (FOB, $/t)']]
    d1.columns = [f"HRC (FOB, $/t)"]
    d1["t"] = f"{x} HRC (FOB, $/t)"
    d2 = df_forecast_processed[[f"{x} HRC (FOB, $/t)_forecast"]]
    d2.columns = [f"HRC (FOB, $/t)"]
    d2["t"] = f"{x} HRC (FOB, $/t)_forecast"
    d = pd.concat([d1, d2])
    return d

d = [fun(i) for i in country]
d3 = pd.concat(d)

fig = px.line(d3, x=d3.index, y="HRC (FOB, $/t)", color="t", 
                markers=False, color_discrete_sequence=['#0E549B', 'red', '#FFCE44', 'violet'])
fig.update_traces(hovertemplate='%{y}')
fig.update_layout(title = {'text': "/".join(country)+" Forecasting HRC prices", 'x': 0.5, 'y': 0.96, 'xanchor': 'center'},
                  margin = dict(t=30), height = 500,
                  legend=dict(title="", yanchor="top", y=0.99, xanchor="center", x=0.5, orientation="h"),
                  xaxis={"title":"Date"},
                  paper_bgcolor='rgba(0, 0, 0, 0)', plot_bgcolor='rgba(0, 0, 0, 0)')

col[1].plotly_chart(fig, use_container_width=True, height=400)

if ff:
    col[0].dataframe(final_df, height=406)
else:
    col[0].dataframe(final_df, height=390)

# Add Landed Price Calculation Tables
st.write("---")
china_col, jk_col = col[1].columns(2)

# Get forecasted FOB prices from final forecast month
forecast_china = df_forecast_processed["China HRC (FOB, $/t)_forecast"].iloc[-1]
forecast_japan = df_forecast_processed["Japan HRC (FOB, $/t)_forecast"].iloc[-1]

with china_col:
    st.subheader("China Import Calculation")
    sea_freight_c = st.number_input("Sea Freight ($/t) - China", value=30.0)
    exchange_rate_c = st.number_input("Exchange Rate (₹/$) - China", value=80.0)
    st.number_input("HRC FOB Price ($/t) - China", value=forecast_china, disabled=True)
    duty_c = 0.075
    lc_port_c = st.number_input("LC + Port Charges ($/t) - China", value=10.0)
    inland_c = st.number_input("Freight Port to City (₹/t) - China", value=500)
    landed_price_c = exchange_rate_c * (lc_port_c + 1.01 * (forecast_china + sea_freight_c) * (1 + 1.1 * duty_c)) + inland_c
    st.metric("China HRC Landed Price (₹/t)", f"{landed_price_c:,.0f}")

with jk_col:
    st.subheader("Japan/Korea Import Calculation")
    sea_freight_j = st.number_input("Sea Freight ($/t) - Japan/Korea", value=30.0)
    exchange_rate_j = st.number_input("Exchange Rate (₹/$) - Japan/Korea", value=80.0)
    st.number_input("HRC FOB Price ($/t) - Japan/Korea", value=forecast_japan, disabled=True)
    duty_j = 0.0
    lc_port_j = st.number_input("LC + Port Charges ($/t) - Japan/Korea", value=10.0)
    inland_j = st.number_input("Freight Port to City (₹/t) - Japan/Korea", value=500)
    landed_price_j = exchange_rate_j * (lc_port_j + 1.01 * (forecast_japan + sea_freight_j) * (1 + 1.1 * duty_j)) + inland_j
    st.metric("Japan/Korea HRC Landed Price (₹/t)", f"{landed_price_j:,.0f}")
