import pandas as pd

df = pd.read_csv('/home/rgukt/HousePricePrediction_MultilinearRegression/data/redfin_housing.csv')

print(df.head())

# print(df.columns)

# print(df.isnull().sum())

# print(df['City'].unique())

# print(df['Date'])

# 3. Extract feature from Date before dropping it
df['Date'] = pd.to_datetime(df['Date'])
df['Sale_Year'] = df['Date'].dt.year
df['Sale_Month'] = df['Date'].dt.month

# 4. Drop unneeded text columns
df = df.drop(columns=['Address', 'Date'])

# # 2. Impute missing feature values with median
df['Beds'] = df['Beds'].fillna(df['Beds'].median())
df['Baths'] = df['Baths'].fillna(df['Baths'].median())

# print(df.head())

# 1. Kwentaen ti overall mean price iti train dataset
overall_mean = df['Price (USD)'].mean()

# 2. Kwentaen ti mean price tunggal siudad
city_price_map = df.groupby('City')['Price (USD)'].mean().to_dict()

# 3. I-map dagiti pabalor ti 'City', ken fillingan dagiti baro/awan a siudad iti overall mean
df['City_Avg_Price'] = df['City'].map(city_price_map).fillna(overall_mean)

# 4. Idiaya wenno ikkatang ti orihinal a 'City' column
df = df.drop(columns=['City'])

# 1. Drop rows missing the target variable (Price)
df = df.dropna(subset=['Price (USD)'])

print(df.head())

# Save cleaned dataframe to data/ folder
df.to_csv('data/cleaned_redfin_housing.csv', index=False)