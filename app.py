import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
import plotly.graph_objects as go

st.title('Stock Forecast App')

stocks = ('GOOG', 'AAPL', 'MSFT', 'AMZN','META','TSLA','BRK.A','JNJ','KO','LLY')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

# User defined parameters
years = st.slider('Years of prediction:', 1, 10)  
period = years * 365 # days

# Load data
start_date = '2010-01-01' 
today = date.today().strftime("%Y-%m-%d")

data = yf.download(selected_stock, start_date, today)

# Preview data
st.subheader('Raw data')
st.write(data.tail())

# Train Prophet model
df_train = data[['Close']]
df_train = df_train.reset_index()
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"}) 

model = Prophet()
model.fit(df_train)

# Make prediction
future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())
    
st.write(f'Forecast plot for {years} years')
fig1 = model.plot(forecast)
st.write(fig1)

st.write(f'Forecast components')
fig2 = model.plot_components(forecast)
st.write(fig2)
