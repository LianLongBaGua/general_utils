from datetime import datetime, timedelta

import pandas as pd 
import numpy as np

from elite_database import Database
from vnpy.trader.constant import Interval, Exchange
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn


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


def create_sequence_all(df, seq_length):
    """
    create list of all col price sequences of length 'seq_length'
    """
    x_cols = [col for col in df.columns if col != f'-{seq_length}_move_label']
    xs = np.stack([df[x_col].values for x_col in x_cols], axis=-1)
    xss = np.stack([xs[i:i+seq_length] for i in range(xs.shape[0]-seq_length-1)], axis=0)
    ys = np.vstack(df[f'-{seq_length}_move_label'].iloc[seq_length:-1].values)

    return np.array(xss), np.array(ys)


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


def prep_for_dl(df: pd.DataFrame, lag: int):
    """prepare dataset"""
    standardize(df, [], True)
    df['open_interest'] = df['open_interest'].pct_change()
    df['volume'] = df['volume'].pct_change()
    df.dropna(inplace=True)
    prep_label(df, 'close_price', -lag, 'one_hot')
    X, y = create_sequence_all(df, lag)
    X, y = torch.tensor(X, dtype=torch.float32), torch.Tensor(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=32)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return train_loader, test_loader


def prep_for_backtest(df: pd.DataFrame, lag: int):
    """prepare dataset"""
    standardize(df, [], True)
    df['open_interest'] = df['open_interest'].pct_change()
    df['volume'] = df['volume'].pct_change()
    df.dropna(inplace=True)
    df.drop(columns='turnover', inplace=True)
    prep_label(df, 'close_price', -lag, 'one_hot')
    X, y = create_sequence_all(df, lag)
    X, y = torch.tensor(X, dtype=torch.float32), torch.Tensor(y)

    return X, y


def predict(model, X, y):
    """predict and return results"""
    y_pred = model(X)
    y_pred = torch.argmax(y_pred, dim=1)
    y_pred = y_pred.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    print(classification_report(y, y_pred))
    print(confusion_matrix(y, y_pred))
    print(accuracy_score(y, y_pred))
    print(f1_score(y, y_pred, average='weighted'))