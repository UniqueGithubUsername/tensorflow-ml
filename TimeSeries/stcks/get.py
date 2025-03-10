import datetime
import yfinance as yf
import pandas as pd

# Adobe, Airbnb, Alphabet A, Alphabet C, Amazon, AMD, Apple, ASML, Broadcom, eBay, Intel, Meta, Microsoft, Netflix, Nvidia, Palantir, Paypal, Qualcomm, Tesla

# tickers
tickers = yf.Tickers('AAPL')
#tickers_array = ['ADBE', 'GOOG', 'GOOGL', 'AMZN', 'AMD', 'AAPL', 'ASML', 'AVGO', 'EBAY' , 'INTC', 'META', 'MSFT', 'NFLX', 'NVDA', 'PLTR', 'PYPL', 'QCOM', 'TSLA']
tickers_array = ['GOOG', 'GOOGL', 'AMZN', 'AMD', 'AAPL', 'ASML', 'META', 'MSFT', 'NFLX', 'NVDA', 'TSLA']

# dates
start_date = datetime.datetime.strptime('2020-01-01', '%Y-%m-%d')
end_date = datetime.date.today()

print("Start date:")
print(start_date)
print("End date:")
print(end_date)

df = yf.download(tickers_array, start=start_date, end=end_date, interval='1d')

# flatten index
df = df.reset_index()
df.columns = df.columns.to_flat_index()

# output data
print("Index:")
print(df.index)

print("Columns:")
print(df.columns)

print("Data:")
print(df)

# save to .csv
df.to_csv('data3.csv')