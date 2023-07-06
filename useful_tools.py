import pandas as pd 
import numpy as np


def prep_label_pct(df, col: str, shift: int, noise=0.01):
    """
    compare whether the next nth bar is an up or a down move
    label using one_hot encoding
    up = [1, 0, 0]
    down = [0, 0, 1]
    noise = [0, 1, 0]
    """
    def compare_values(row, column1: str, column2: str, noise=noise):
        percent_diff = (row[column2] - row[column1]) / row[column1]

        if percent_diff > noise:
            return [1, 0, 0]
        elif percent_diff < -noise:
            return [0, 0, 1]
        else:
            return [0, 1, 0]

    df[col + "_shifted"] = df[col].shift(shift)

    df[f'{shift}_move_label'] = df.apply(compare_values, axis=1, args=(col, col + "_shifted"))


def trim_df(df, keep_symbol=False):
    """Trim everything other than OHLCV with option to keep symbol"""
    df.drop(['exchange', 'interval', 'datetime', 'gateway_name'], axis=1, inplace=True)
    
    if not keep_symbol:
        df.drop(['symbol'], axis=1, inplace=True)