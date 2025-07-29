"""
Feature engineering utilities for Amazon Food Recommender.
"""
import pandas as pd

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df['ReviewTime'] = pd.to_datetime(df['Time'], unit='s')
    df['Year'] = df['ReviewTime'].dt.year
    df['Month'] = df['ReviewTime'].dt.month
    return df

def filter_users_items(df: pd.DataFrame, min_user=5, min_item=10) -> pd.DataFrame:
    df1 = df[df.groupby('UserId').UserId.transform('size') >= min_user]
    df2 = df1[df1.groupby('ProductId').ProductId.transform('size') >= min_item]
    return df2
