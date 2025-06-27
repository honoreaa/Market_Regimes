

#1. calculate daily returns, 
#2. calclate 10-day volatility
#3. drop all rows w missing vals

import pandas as pd

def add_features(df):
    """
    Adds daily returns and 10-day rolling volatility to the dataframe.
    Drops NA values created by rolling.
    """
    df = df.copy()
    df['return'] = df['Close'].pct_change()
    df['volatility'] = df['return'].rolling(window=10).std()
    df = df.dropna()
    return df
