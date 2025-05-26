import yfinance as yf

dat = yf.Ticker("ONDS")
data = dat.history(period='5y', interval='1d')

# Select features
features = ['Open','High','Low','Close','Volume']
data = data[features]

print(data.head())
print(data.columns)

data.to_csv('data/ONDS-days.csv')