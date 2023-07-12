from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import torch

from elite_database import Database
from vnpy.trader.constant import Interval, Exchange


def round_to_ones(pred):
    """
    round the input tensor to the nearest integer
    """
    max_indices = pred.argmax(dim=1)
    one_hot = torch.zeros_like(pred)
    one_hot.scatter_(1, max_indices.unsqueeze(1), 1)

    return one_hot


def load_essentials(symbol: str, start: str, end: str, exchange: str):
    """
    Load data from database with end=now and interval=1m
    returns df with columns:
        open_price, high_price, low_price,
        close_price, volume, open_interest, turnover
    """
    db = Database()

    df = db.load_bar_df(
        symbol,
        Exchange(exchange),
        Interval.MINUTE,
        datetime.strptime(start, "%Y-%m-%d"),
        datetime.strptime(end, "%Y-%m-%d"),
    )
    trim_df(df)

    return df


def prep_label(df, col: str, forward_window: int, label: str, noise=0.01):
    """
    compare whether the next nth bar is an up or a down move
    label using one_hot encoding
    up = [1, 0, 0]
    down = [0, 0, 1]
    noise = [0, 1, 0]
    label using binary encoding
    up = 1
    down = -1
    noise = 0
    rows with NaN values are dropped
    this function doesn't return anything, it modifies the dataframe in place
    """

    def compare_values(row, column1: str, column2: str, noise=noise):
        percent_diff = (row[column2] - row[column1]) / row[column1]

        if label == "one_hot":
            if percent_diff > noise:
                return np.array([1, 0, 0])
            elif percent_diff < -noise:
                return np.array([0, 0, 1])
            else:
                return np.array([0, 1, 0])
        elif label == "binary":
            if percent_diff > noise:
                return 1
            elif percent_diff < -noise:
                return -1
            else:
                return 0

    df[col + "_shifted"] = df[col].shift(forward_window)

    df[f"{forward_window}_move_label"] = df.apply(
        compare_values, axis=1, args=(col, col + "_shifted")
    )

    df.dropna(inplace=True)
    df.drop([col + "_shifted"], axis=1, inplace=True)


def create_sequence_all(df, lag_window):
    arr = df.values
    resultX = np.empty((len(arr) - lag_window + 1, lag_window, arr.shape[1]-1))
    resulty = np.empty((len(arr) - lag_window + 1, 3))
    for start in range(len(arr) - lag_window + 1):
        resultX[start] = arr[start : start + lag_window, 0:-1]
        resulty[start] = arr[start + lag_window-1, -1]
    
    return resultX, resulty


def trim_df(df: pd.DataFrame, keep_symbol=False):
    """Trim everything other than OHLCV with option to keep symbol"""
    df.drop(
        columns=["exchange", "interval", "turnover", "datetime", "gateway_name"],
        axis=1,
        inplace=True,
    )

    if not keep_symbol:
        df.drop(["symbol"], axis=1, inplace=True)


def divide_means(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("The lists must have the same length.")

    # Calculate the means of the arrays in each list.

    means1 = [np.mean(array) for array in list1]
    means2 = [np.mean(array) for array in list2]

    # Divide the means of the arrays in list1 by the means of the arrays in list2.

    return [mean1 / mean2 for mean1, mean2 in zip(means1, means2)]


def evaluate_signal(pred) -> int:
    if pred == torch.tensor[1., 0., 0.]:
        return 1
    elif pred == torch.tensor[0., 1., 0.]:
        return 0
    else:
        return -1


def renaming(df: pd.DataFrame):
    """rename columns"""
    df.rename(
        columns={
            "close_price": "close",
            "high_price": "high",
            "low_price": "low",
            "open_price": "open",
        },
        inplace=True,
    )
    return df
