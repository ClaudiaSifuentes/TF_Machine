import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def process(df, scale = True, scaler=None):
    if('Source' in df.columns):
        df.drop(['Source'], axis=1, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Timestamp'] = df['Date'].apply(lambda x: x.timestamp())
    df['high_low_diff'] = df['High'] - df['Low']
    df['open_close_diff'] = df['Close'] - df['Open']
    df.dropna(inplace=True);
    df.drop(['Date'], axis=1, inplace=True)
    df.sort_values('Timestamp', inplace=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    if scale:
        X = scaler.transform(df.values)
    else:
        X = df.values
    return X

def to_number(df):
    df['High'] = df['High'].astype(float)
    df['Low'] = df['Low'].astype(float)
    df['Open'] = df['Open'].astype(float)
    df['Close'] = df['Close'].astype(float)
    df['Volume'] = df['Volume'].astype(float)
    df['Market Cap'] = df['Market Cap'].astype(float)
    return df