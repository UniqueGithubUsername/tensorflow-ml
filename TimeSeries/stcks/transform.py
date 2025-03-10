import pandas as pd
import numpy as np

df=pd.read_csv('data3.csv')

# remove unnamed column
df.pop('Unnamed: 0')
df = df.fillna(0)

# output data
print("Index:")
print(df.index)

print("Columns:")
print(df.columns)

print("Data:")
print(df)

df = df.set_index("('Date', '')").diff()
df = df.fillna(0)

print("Data:")
print(df)

for col in df.columns:
	print(col)
	df[col] = np.where(df[col] < 0, -1, 1)

print("Data:")
print(df)

df.to_csv('data_transformed.csv')