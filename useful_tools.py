import pandas as pd 
import numpy as np

from datetime import datetime, timedelta

from elite_database import Database
from vnpy.trader.constant import Interval, Exchange
from sklearn.preprocessing import StandardScaler


def load_essentials(symbol: str, start: str, end: str, exchange: str):
    """Load data from database with end=now and interval=1m"""
    db = Database()
    df = db.load_bar_df(
        symbol, Exchange(exchange), 
        Interval.MINUTE, 
        datetime.strptime(start, '%Y-%m-%d'), 
        datetime.strptime(end, '%Y-%m-%d')
        )

    trim_df(df)

    return df


def prep_label(df, col: str, shift: int, label: str, noise=0.01):
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
    """
    def compare_values(row, column1: str, column2: str, noise=noise):
        percent_diff = (row[column2] - row[column1]) / row[column1]

        if label == 'one_hot':
            if percent_diff > noise:
                return np.array([1, 0, 0])
            elif percent_diff < -noise:
                return np.array([0, 0, 1])
            else:
                return np.array([0, 1, 0])
        elif label == 'binary':
            if percent_diff > noise:
                return 1
            elif percent_diff < -noise:
                return -1
            else:
                return 0

    df[col + "_shifted"] = df[col].shift(shift)

    df[f'{shift}_move_label'] = df.apply(compare_values, axis=1, args=(col, col + "_shifted"))

    df.dropna(inplace=True)


def create_sequences(df, col, seq_length):
    """
    create list of 'col' price sequences of length 'seq_length'
    TODO: add option to create sequences of multiple columns
    """
    xs = []
    ys = []

    for i in range(df.shape[0]-seq_length-1):
        x = df[col].iloc[i:(i+seq_length)]
        xs.append(x)
        y = df[f'-{seq_length}_move_label'].iloc[i+seq_length]
        ys.append(y)

    return np.array(xs), np.array(ys)


def trim_df(df: pd.DataFrame, keep_symbol=False):
    """Trim everything other than OHLCV with option to keep symbol"""
    df.drop(columns=['exchange', 'interval', 'turnover', 'datetime', 'gateway_name'], axis=1, inplace=True)
    
    if not keep_symbol:
        df.drop(['symbol'], axis=1, inplace=True)


def standardize(df, cols: list = [], scale_all: bool = False):
    """
    Standardize columns
    Use the same standardization for all columns except volume and open_interest
    """
    scaler = StandardScaler()

    if scale_all:
        cols = df.columns.tolist()
        cols.remove('volume')
        cols.remove('open_interest')
        df[cols] = scaler.fit_transform(df[cols])
        df['volume'] = scaler.fit_transform(df['volume'].values.reshape(-1, 1))
        df['open_interest'] = scaler.fit_transform(df['open_interest'].values.reshape(-1, 1))
    else:
        for col in cols:
            df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))


def divide_means(list1, list2):

    if len(list1) != len(list2):
        raise ValueError("The lists must have the same length.")

      # Calculate the means of the arrays in each list.

    means1 = [np.mean(array) for array in list1]
    means2 = [np.mean(array) for array in list2]

      # Divide the means of the arrays in list1 by the means of the arrays in list2.

    return [mean1 / mean2 for mean1, mean2 in zip(means1, means2)]