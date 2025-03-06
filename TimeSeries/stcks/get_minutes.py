import datetime
import yfinance as yf
import pandas as pd

# tickers
tickers = yf.Tickers('AAPL')

# dates
#start_date = datetime.date.today() - datetime.timedelta(days=29)
#end_date = start_date + datetime.timedelta(days=8)

start_date = datetime.datetime.strptime('2020-01-01', '%Y-%m-%d')
end_date = datetime.date.today()

df_res = pd.DataFrame()

for i in range(1):
	print("Durchlauf: " , i)

	print("Start date:")
	print(start_date)
	print("End date:")
	print(end_date)

	df = yf.download(['AAPL'], period='8d', interval='1d', start=start_date, end=end_date)

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

	start_date = end_date
	end_date = end_date + datetime.timedelta(days=8)

	df_res = pd.concat([df_res, df], ignore_index=True)


print(df_res)
# save to .csv
#df.to_csv('data.csv')