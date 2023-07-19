from datetime import datetime
import itertools

import pandas as pd
import numpy as np

from elite_database import Database
from vnpy.trader.constant import Interval, Exchange
from elite_ctastrategy import HistoryManager

from hurst import compute_Hc


def resample(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """K线合成"""
    df_resampled = df.resample(interval).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    })

    df_resampled = df_resampled.drop_duplicates()
    df.fillna(method='pad', inplace=True)

    return df_resampled


def calculate_periodic_hurst(series: np.array, period: int, step: int = 1):
    """
    calculate the Hurst exponent for a given period in a rolling manner
    """
    hurst = []
    for i in range(0, len(series) - period, step):
        hurst.append(compute_Hc(series[i : i + period], kind="price")[0])

    return hurst


def calculate_periodic_efficiency_ratio(series: np.array, period: int, step: int = 1):
    """
    calculate the efficiency ratio for a given period in a rolling manner
    """
    er = []
    for i in range(0, len(series) - period, step):
        er.append(
            (series[i + period] - series[i])
            / (max(series[i : i + period]) - min(series[i : i + period]))
        )
    return er


def calculate_efficiency_ratio(high: np.array, low: np.array, period: int):
    """
    calculate the efficiency ratio for a given period
    """
    ind_changes = high[-period:] - low[-period:]
    sum_of_ind_changes = abs(ind_changes).sum()
    if sum_of_ind_changes < 0.0001:
        return 1
    assert sum_of_ind_changes != 0., "sum of individual changes cannot be zero"
    er_value = (high[-1] - low[-(period+1)]) / sum_of_ind_changes
    return abs(er_value)


def calculate_eff(high: np.array, low: np.array, period: int):
    one_period_diff: np.array = high[-period:] - low[-period:]
    sum_of_one_period_diff: float = one_period_diff.sum()
    nday_diff = high[-1] - low[-(period+1)]
    return abs(nday_diff) / sum_of_one_period_diff

def round_to_ones(pred):
    """
    round the input tensor to the nearest integer
    """
    max_indices = pred.argmax(dim=1)
    one_hot = np.zeros_like(pred)
    one_hot.scatter_(1, max_indices.unsqueeze(1), 1)

    return one_hot


def standardize(df: pd.DataFrame):
    """
    Standardize columns
    Use the same standardization for OHLC
    The rest are standardized separately
    Use StandardScaler as it is more robust to outliers
    """
    for col in df.columns:
        scaler = StandardScaler()
        df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))


def load_essentials(symbol: str, start: str, end: str, exchange: str):
    """Load data from database with interval=1m"""
    db = Database()
    df = db.load_bar_df(
        symbol,
        Exchange(exchange),
        Interval.MINUTE,
        datetime.strptime(start, "%Y-%m-%d"),
        datetime.strptime(end, "%Y-%m-%d"),
    )
    trim_df(df)

    return renaming(df)


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
    resultX = np.empty((len(arr) - lag_window + 1, lag_window, arr.shape[1] - 1))
    resulty = np.empty((len(arr) - lag_window + 1, 3))
    for start in range(len(arr) - lag_window + 1):
        resultX[start] = arr[start : start + lag_window, 0:-1]
        resulty[start] = arr[start + lag_window - 1, -1]

    return resultX, resulty


def create_sequences(df, col, seq_length):
    """
    create list of 'col' price sequences of length 'seq_length'
    TODO: add option to create sequences of multiple columns
    """
    xs = []
    ys = []

    for i in range(df.shape[0] - seq_length - 1):
        x = df[col].iloc[i : (i + seq_length)]
        xs.append(x)
        y = df[f"-{seq_length}_move_label"].iloc[i + seq_length]
        ys.append(y)

    return np.array(xs), np.array(ys)


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


def evaluate_one_hot_signal(pred) -> int:
    """evaluate one_hot signal"""
    if pred == np.array[1.0, 0.0, 0.0]:
        return 1
    elif pred == np.array[0.0, 1.0, 0.0]:
        return 0
    else:
        return -1


def evaluate_str_signal(pred) -> int:
    """evaluate str signal"""
    if pred == "up":
        return 1
    elif pred == "neutral":
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


def remove_infs_and_zeros(df: pd.DataFrame):
    """remove inf values and zeros"""
    df.replace([np.inf, -np.inf, np.zeros], 0.0001, inplace=True)
    return df


def calc_nperiod_change(df: pd.DataFrame, feature: str, n: int):
    """calculate n period change"""
    import warnings
    warnings.filterwarnings("ignore")
    df[f"{n}_period_{feature}_change"] = df[feature].diff(n).fillna(0)


def get_feature_corr(
    df: pd.DataFrame, feature: str, start, end, step, method="pearson"
):
    """get feature correlation"""
    corr = pd.DataFrame()
    df_forward, df_backward = df.copy(), df.copy()
    for i in range(start, end, step):
        calc_nperiod_change(df_forward, feature, -i)
        calc_nperiod_change(df_backward, feature, i)
    df_conc = pd.concat([df_forward, df_backward], axis=1)
    df_conc = drop_ohlcv_cols(df_conc)
    df_conc = df_conc.drop([feature], axis=1)
    corr = df_conc.corr(method=method)
    for i in range(start, end, step):
        corr = corr.drop([f"{i}_period_{feature}_change"], axis=1)
        corr = corr.drop([f"{-i}_period_{feature}_change"], axis=0)
    return corr


def calc_nperiod_returns(df: pd.DataFrame, n: int):
    """calculate n period returns"""
    import warnings
    warnings.filterwarnings("ignore")
    df[f"{n}_period_returns"] = df["close"].pct_change(n).fillna(0)


def calc_nperiod_volatility_backward(df: pd.DataFrame, n: int):
    """calculate n period volatility"""
    import warnings
    warnings.filterwarnings("ignore")
    df[f"{n}_period_volatility_backward"] = (
        df["close"].pct_change().rolling(n).std().fillna(0)
    )


def calc_nperiod_volatility_forward(df: pd.DataFrame, n: int):
    """calculate n period forward volatility"""
    df = df.copy()
    df[f"{n}_period_volatility_forward"] = (
        df[f"{n}_period_volatility_backward"].shift(-n).fillna(0)
    )
    return df[f"{n}_period_volatility_forward"]


def calc_volatility_change(df: pd.DataFrame, n: int):
    """calculate volatility change"""
    df[f"{n}_period_volatility_change"] = (
        df[f"{n}_period_volatility_backward"].pct_change(n).fillna(0)
    )
    return df[f"{n}_period_volatility_change"]


def drop_ohlcv_cols(df: pd.DataFrame):
    """drop ohlcv columns"""
    return df.drop(
        columns=["open", "high", "low", "close", "volume", "open_interest"], axis=1
    )


def get_return_corr(
    df: pd.DataFrame, start: int, end: int, step: int, method="pearson"
):
    """get return correlation"""
    corr = pd.DataFrame()
    df_forward, df_backward = df.copy(), df.copy()
    for i in range(start, end, step):
        calc_nperiod_returns(df_forward, -i)
        calc_nperiod_returns(df_backward, i)
    df_conc = pd.concat([df_forward, df_backward], axis=1)
    df_conc = drop_ohlcv_cols(df_conc)
    corr = df_conc.corr(method=method)
    for i in range(start, end, step):
        corr = corr.drop([f"{i}_period_returns"], axis=1)
        corr = corr.drop([f"{-i}_period_returns"], axis=0)
    return corr


def get_volatility_corr(
    df: pd.DataFrame, start: int, end: int, step: int, method="pearson"
):
    """get volatility correlation"""
    corr = pd.DataFrame()
    df_forward, df_backward = df.copy(), df.copy()
    for i in range(start, end, step):
        calc_nperiod_volatility_backward(df_backward, i)
        df_forward = df_forward.join(calc_nperiod_volatility_forward(df_backward, i))
    df_conc = pd.concat([df_forward, df_backward], axis=1)
    df_conc = drop_ohlcv_cols(df_conc)
    corr = df_conc.corr(method=method)
    for i in range(start, end, step):
        corr = corr.drop([f"{i}_period_volatility_backward"], axis=1)
        corr = corr.drop([f"{i}_period_volatility_forward"], axis=0)
    return corr


# def get_volatility_change_corr(df: pd.DataFrame, start: int, end: int, n: int, interval: int):
#     """get volatility change correlation"""
#     corr = pd.DataFrame()
#     df_forward, df_backward = df.copy(), df.copy()
#     for i in range(start, end, interval):
#         calc_nperiod_volatility_backward(df_backward, interval)
#         calc_volatility_change(df_backward, n)
#         df_forward.join(calc_nperiod_volatility_forward(df_backward, i))
#     df_conc = pd.concat([df_forward, df_backward], axis=1)
#     df_conc = drop_ohlcv_cols(df_conc)
#     corr = df_conc.corr()
#     for i in range(start, end, interval):
#         corr = corr.drop([f"{i}_period_volatility_backward"], axis=1)
#         corr = corr.drop([f"{i}_period_volatility_forward"], axis=0)
#     return corr
