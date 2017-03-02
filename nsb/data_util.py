import pandas as pd


def columns(df):
    return df.columns.values.tolist()


def category_columns(df):
    return [c for c in columns(df) if str(df[c].dtypes) == "category"]

