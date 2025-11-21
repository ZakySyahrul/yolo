import pandas as pd

def get_linechart_data(df):
    # Kolom pertama selalu adalah kolom waktu (frame/second/minute/hour)
    time_col = df.columns[0]

    return df.groupby(time_col)["total"].sum().reset_index(name="count")


def get_barchart_data(df):
    type_cols = [c for c in df.columns if c.startswith("type_")]
    return df[type_cols].sum()

def get_density_counts(df):
    return df["density"].value_counts()
