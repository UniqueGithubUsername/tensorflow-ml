from datetime import date, timedelta
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

output_filename = 'data_minutes'
tickers_string = 'NVDA TSLA AAPL MSFT PLTR AMZN GOOGL META SOFI AMD IBM AVGO JPM KO CSCO CVX JNJ PEP MCD WMT MA HOOD BLK SBUX GD PG SMCI NUE COIN CAH'
tickers_array = tickers_string.split(' ')
tickers = yf.Tickers(tickers_string)
steps = 4

end = date.today()
start = end - timedelta(days=7)
print(end)
print(start)
data = pd.DataFrame()
for i in range(steps):
    data = pd.concat([yf.download(tickers_array, interval='1m', start=start, end=end, multi_level_index=False), data])
    print(data)
    end = start
    start = end - timedelta(days=7)
    print(end)
    print(start)

# Build normal index columns from multi index columns
new_columns = []
for col in data.columns:
    new_columns.append(col[1] + '_' + col[0])

# Exchange to normal index columns
data.columns = new_columns

# Save unprocessed data
data.to_csv(output_filename+'_'+start.strftime('%Y-%m-%d')+'_'+end.strftime('%Y-%m-%d')+'.csv')

# Transform and normalize Data
data_formatted = pd.DataFrame()
data_formatted.index = data.index

for ticker in tickers_array:
    # Percentages
    data_formatted[ticker+"_close_pct"] = data[ticker+"_Close"].pct_change()
    data_formatted[ticker+"_open_pct"] = data[ticker+"_Open"].pct_change()
    data_formatted[ticker+"_high_pct"] = data[ticker+"_High"].pct_change()
    data_formatted[ticker+"_low_pct"] = data[ticker+"_Low"].pct_change()
    data_formatted[ticker+"_volume_pct"] = data[ticker+"_Volume"].pct_change()
    # Relations
    data_formatted[ticker+"_change_day_pct"] = data[ticker+"_Close"] / data[ticker+"_Open"]
    data_formatted[ticker+"_high_low_pct"] = data[ticker+"_High"] / data[ticker+"_Low"]
    data_formatted[ticker+"_open_high_pct"] = data[ticker+"_Open"] / data[ticker+"_High"]
    data_formatted[ticker+"_open_low_pct"] = data[ticker+"_Open"] / data[ticker+"_Low"]
    data_formatted[ticker+"_high_close_pct"] = data[ticker+"_High"] / data[ticker+"_Close"]
    data_formatted[ticker+"_low_close_pct"] = data[ticker+"_Low"] / data[ticker+"_Close"]
    # Overnight
    data_formatted[ticker+"_change_overnight_pct"] = data[ticker+"_Open"] / data[ticker+"_Close"].shift(1)
    #print(data[ticker+"_Close"].shift(1))

data_formatted = data_formatted.fillna(0)

data_formatted.to_csv(output_filename+'_'+start.strftime('%Y-%m-%d')+'_'+end.strftime('%Y-%m-%d')+'_formatted.csv')