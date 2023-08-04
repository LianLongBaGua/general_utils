from datetime import datetime
import warnings

import pandas as pd
import numpy as np
import talib
from talib import abstract

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
        "volume": "sum",
        "open_interest": "last"
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


def round_to_ones(pred):
    """
    round the input tensor to the nearest integer
    """
    max_indices = pred.argmax(dim=1)
    one_hot = np.zeros_like(pred)
    one_hot.scatter_(1, max_indices.unsqueeze(1), 1)

    return one_hot


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


def calculate_indicators(df, group_number, default_lookback=14):
    # Split the functions into 5 groups
    all_functions = talib.get_functions()
    group_size = len(all_functions) // 100
    function_groups = [all_functions[i:i+group_size] for i in range(0, len(all_functions), group_size)]
    
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    for name in function_groups[group_number]:
        try:
            indicator = getattr(abstract, name)

            # Set up the default inputs for the function
            inputs = {
                'open': df['open'],
                'high': df['high'],
                'low': df['low'],
                'close': df['close'],
                'volume': df['volume'],
                'timeperiod': default_lookback,
            }

            # Call the function with the inputs, ignoring any arguments it doesn't need
            output = indicator(inputs, **{key: inputs[key] for key in indicator.parameters.keys()})
            
            if output is None:
                continue

            # If the function returns a single Series, add it to the DataFrame directly
            if isinstance(output, pd.core.series.Series):
                df[name] = output
            # If the function returns a list of Series, add each one to the DataFrame
            else:
                for i, result in enumerate(output):
                    df[f'{name}_{i}'] = result

        except Exception as e:
            print(f"Error with {name}: {e}")
            continue
    return df

def prepare(hm):
    """
    Prepare the technical indicators
    """
    hm['RSI7'] = talib.RSI(hm.close, 7) 
    hm['RSI12'] = talib.RSI(hm.close, 12) 
    hm['RSI30'] = talib.RSI(hm.close, 30)
    hm['RSI50'] = talib.RSI(hm.close, 50)
    hm['RSI75'] = talib.RSI(hm.close, 75)
    hm['RSI100'] = talib.RSI(hm.close, 100)
    hm['ROC25'] = talib.ROC(hm.close, 25)
    hm['ROC50'] = talib.ROC(hm.close, 50)
    hm['ROC75'] = talib.ROC(hm.close, 75)
    hm['ROC100'] = talib.ROC(hm.close, 100)
    hm['MOM30'] = talib.MOM(hm.close, 30)
    hm['MOM50'] = talib.MOM(hm.close, 50)
    hm['MOM75'] = talib.MOM(hm.close, 75)
    hm['MOM100'] = talib.MOM(hm.close, 100)
    hm['PLUSDM30'] = talib.PLUS_DM(hm.high, hm.low, 30)
    hm['PLUSDM50'] = talib.PLUS_DM(hm.high, hm.low, 50)
    hm['PLUSDM75'] = talib.PLUS_DM(hm.high, hm.low, 75)
    hm['PLUSDM100'] = talib.PLUS_DM(hm.high, hm.low, 100)
    hm['MFI25'] = talib.MFI(hm.high, hm.low, hm.close, hm.volume, 20)
    hm['MFI50'] = talib.MFI(hm.high, hm.low, hm.close, hm.volume, 50)
    hm['MFI75'] = talib.MFI(hm.high, hm.low, hm.close, hm.volume, 75)
    hm['MFI100'] = talib.MFI(hm.high, hm.low, hm.close, hm.volume, 100)

    _, hm['DEA'], hm['MACD'] = talib.MACD(hm.close, 12, 26, 9)

    hm['NATR14'] = talib.NATR(hm.high, hm.low, hm.close, 14)
    hm['NATR30'] = talib.NATR(hm.high, hm.low, hm.close, 30)
    hm['NATR50'] = talib.NATR(hm.high, hm.low, hm.close, 50)
    hm['NATR75'] = talib.NATR(hm.high, hm.low, hm.close, 75)
    hm['NATR100'] = talib.NATR(hm.high, hm.low, hm.close, 100)

    hm['KELTNER14'] = (hm.close - talib.SMA(hm.close, 14)) / hm['ATR14']
    hm['KELTNER30'] = (hm.close - talib.SMA(hm.close, 30)) / hm['ATR30']
    hm['KELTNER50'] = (hm.close - talib.SMA(hm.close, 50)) / hm['ATR50']
    hm['KELTNER75'] = (hm.close - talib.SMA(hm.close, 75)) / hm['ATR75']
    hm['KELTNER100'] = (hm.close - talib.SMA(hm.close, 100)) / hm['ATR100']

    hm['ULTOSC'] = talib.ULTOSC(hm.high, hm.low, hm.close, 7, 14, 28)
    hm['WILLR14'] = talib.WILLR(hm.high, hm.low, hm.close, 14)
    hm['STOCHRSI14'] = talib.STOCHRSI(hm.close, 14, 3, 3)
    hm['STOCHRSI30'] = talib.STOCHRSI(hm.close, 30, 3, 3)
    hm['STOCHRSI50'] = talib.STOCHRSI(hm.close, 50, 3, 3)
    hm['STOCHRSI75'] = talib.STOCHRSI(hm.close, 75, 3, 3)
    hm['STOCHRSI100'] = talib.STOCHRSI(hm.close, 100, 3, 3)

    hm['OBV'] = talib.OBV(hm.close, hm.volume)
    hm['ADOSC'] = talib.ADOSC(hm.high, hm.low, hm.close, hm.volume, 3, 10)
    hm['AD'] = talib.AD(hm.high, hm.low, hm.close, hm.volume)

    hm['CDL2CROWS'] = talib.CDL2CROWS(hm.open, hm.high, hm.low, hm.close)
    hm['CDL3BLACKCROWS'] = talib.CDL3BLACKCROWS(hm.open, hm.high, hm.low, hm.close)
    hm['CDL3INSIDE'] = talib.CDL3INSIDE(hm.open, hm.high, hm.low, hm.close)
    hm['CDL3LINESTRIKE'] = talib.CDL3LINESTRIKE(hm.open, hm.high, hm.low, hm.close)
    hm['CDL3OUTSIDE'] = talib.CDL3OUTSIDE(hm.open, hm.high, hm.low, hm.close)
    hm['CDL3STARSINSOUTH'] = talib.CDL3STARSINSOUTH(hm.open, hm.high, hm.low, hm.close)
    hm['CDL3WHITESOLDIERS'] = talib.CDL3WHITESOLDIERS(hm.open, hm.high, hm.low, hm.close)
    hm['CDLABANDONEDBABY'] = talib.CDLABANDONEDBABY(hm.open, hm.high, hm.low, hm.close)
    hm['CDLADVANCEBLOCK'] = talib.CDLADVANCEBLOCK(hm.open, hm.high, hm.low, hm.close)
    hm['CDLBELTHOLD'] = talib.CDLBELTHOLD(hm.open, hm.high, hm.low, hm.close)
    hm['CDLBREAKAWAY'] = talib.CDLBREAKAWAY(hm.open, hm.high, hm.low, hm.close)
    hm['CDLCLOSINGMARUBOZU'] = talib.CDLCLOSINGMARUBOZU(hm.open, hm.high, hm.low, hm.close)

    hm['CDLCONCEALBABYSWALL'] = talib.CDLCONCEALBABYSWALL(hm.open, hm.high, hm.low, hm.close)
    hm['CDLCOUNTERATTACK'] = talib.CDLCOUNTERATTACK(hm.open, hm.high, hm.low, hm.close)
    hm['CDLDARKCLOUDCOVER'] = talib.CDLDARKCLOUDCOVER(hm.open, hm.high, hm.low, hm.close)
    hm['CDLDOJI'] = talib.CDLDOJI(hm.open, hm.high, hm.low, hm.close)
    hm['CDLDOJISTAR'] = talib.CDLDOJISTAR(hm.open, hm.high, hm.low, hm.close)
    hm['CDLDRAGONFLYDOJI'] = talib.CDLDRAGONFLYDOJI(hm.open, hm.high, hm.low, hm.close)
    hm['CDLENGULFING'] = talib.CDLENGULFING(hm.open, hm.high, hm.low, hm.close)
    hm['CDLEVENINGDOJISTAR'] = talib.CDLEVENINGDOJISTAR(hm.open, hm.high, hm.low, hm.close)
    hm['CDLEVENINGSTAR'] = talib.CDLEVENINGSTAR(hm.open, hm.high, hm.low, hm.close)
    hm['CDLGAPSIDESIDEWHITE'] = talib.CDLGAPSIDESIDEWHITE(hm.open, hm.high, hm.low, hm.close)
    hm['CDLGRAVESTONEDOJI'] = talib.CDLGRAVESTONEDOJI(hm.open, hm.high, hm.low, hm.close)
    hm['CDLHAMMER'] = talib.CDLHAMMER(hm.open, hm.high, hm.low, hm.close)

    hm['CDLHANGINGMAN'] = talib.CDLHANGINGMAN(hm.open, hm.high, hm.low, hm.close)
    hm['CDLHARAMI'] = talib.CDLHARAMI(hm.open, hm.high, hm.low, hm.close)
    hm['CDLHARAMICROSS'] = talib.CDLHARAMICROSS(hm.open, hm.high, hm.low, hm.close)
    hm['CDLHIGHWAVE'] = talib.CDLHIGHWAVE(hm.open, hm.high, hm.low, hm.close)
    hm['CDLHIKKAKE'] = talib.CDLHIKKAKE(hm.open, hm.high, hm.low, hm.close)
    hm['CDLHIKKAKEMOD'] = talib.CDLHIKKAKEMOD(hm.open, hm.high, hm.low, hm.close)
    hm['CDLHOMINGPIGEON'] = talib.CDLHOMINGPIGEON(hm.open, hm.high, hm.low, hm.close)
    hm['CDLIDENTICAL3CROWS'] = talib.CDLIDENTICAL3CROWS(hm.open, hm.high, hm.low, hm.close)
    hm['CDLINNECK'] = talib.CDLINNECK(hm.open, hm.high, hm.low, hm.close)
    hm['CDLINVERTEDHAMMER'] = talib.CDLINVERTEDHAMMER(hm.open, hm.high, hm.low, hm.close)
    hm['CDLKICKING'] = talib.CDLKICKING(hm.open, hm.high, hm.low, hm.close)
    
    hm['CDLKICKINGBYLENGTH'] = talib.CDLKICKINGBYLENGTH(hm.open, hm.high, hm.low, hm.close)
    hm['CDLLADDERBOTTOM'] = talib.CDLLADDERBOTTOM(hm.open, hm.high, hm.low, hm.close)
    hm['CDLLONGLEGGEDDOJI'] = talib.CDLLONGLEGGEDDOJI(hm.open, hm.high, hm.low, hm.close)
    hm['CDLLONGLINE'] = talib.CDLLONGLINE(hm.open, hm.high, hm.low, hm.close)
    hm['CDLMARUBOZU'] = talib.CDLMARUBOZU(hm.open, hm.high, hm.low, hm.close)
    hm['CDLMATCHINGLOW'] = talib.CDLMATCHINGLOW(hm.open, hm.high, hm.low, hm.close)
    hm['CDLMATHOLD'] = talib.CDLMATHOLD(hm.open, hm.high, hm.low, hm.close)
    hm['CDLMORNINGDOJISTAR'] = talib.CDLMORNINGDOJISTAR(hm.open, hm.high, hm.low, hm.close)
    hm['CDLMORNINGSTAR'] = talib.CDLMORNINGSTAR(hm.open, hm.high, hm.low, hm.close)
    hm['CDLONNECK'] = talib.CDLONNECK(hm.open, hm.high, hm.low, hm.close)
    hm['CDLPIERCING'] = talib.CDLPIERCING(hm.open, hm.high, hm.low, hm.close)
    hm['CDLRICKSHAWMAN'] = talib.CDLRICKSHAWMAN(hm.open, hm.high, hm.low, hm.close)

    hm['CDLRISEFALL3METHODS'] = talib.CDLRISEFALL3METHODS(hm.open, hm.high, hm.low, hm.close)
    hm['CDLSEPARATINGLINES'] = talib.CDLSEPARATINGLINES(hm.open, hm.high, hm.low, hm.close)
    hm['CDLSHOOTINGSTAR'] = talib.CDLSHOOTINGSTAR(hm.open, hm.high, hm.low, hm.close)
    hm['CDLSHORTLINE'] = talib.CDLSHORTLINE(hm.open, hm.high, hm.low, hm.close)
    hm['CDLSPINNINGTOP'] = talib.CDLSPINNINGTOP(hm.open, hm.high, hm.low, hm.close)
    hm['CDLSTALLEDPATTERN'] = talib.CDLSTALLEDPATTERN(hm.open, hm.high, hm.low, hm.close)
    hm['CDLSTICKSANDWICH'] = talib.CDLSTICKSANDWICH(hm.open, hm.high, hm.low, hm.close)
    hm['CDLTAKURI'] = talib.CDLTAKURI(hm.open, hm.high, hm.low, hm.close)
    hm['CDLTASUKIGAP'] = talib.CDLTASUKIGAP(hm.open, hm.high, hm.low, hm.close)
    hm['CDLTHRUSTING'] = talib.CDLTHRUSTING(hm.open, hm.high, hm.low, hm.close)
    hm['CDLTRISTAR'] = talib.CDLTRISTAR(hm.open, hm.high, hm.low, hm.close)
    hm['CDLUNIQUE3RIVER'] = talib.CDLUNIQUE3RIVER(hm.open, hm.high, hm.low, hm.close)
    hm['CDLUPSIDEGAP2CROWS'] = talib.CDLUPSIDEGAP2CROWS(hm.open, hm.high, hm.low, hm.close)
    hm['CDLXSIDEGAP3METHODS'] = talib.CDLXSIDEGAP3METHODS(hm.open, hm.high, hm.low, hm.close)



    hm = hm.dropna()

    return hm

from sklearn.preprocessing import StandardScaler
from pandas_ta import log_return

def prepare_desired_pos(df, lag=50, multiplier=10):
    scaler = StandardScaler()
    df[f'{lag}m_ret'] = scaler.fit_transform(log_return(df.close, length=lag, offset=-lag).values.reshape(-1,1))
    df.dropna(inplace=True)
    df['pos_change'] = df[f'{lag}m_ret'] * multiplier
    df['pos_change'] = df['pos_change'].apply(int)
    df['pos_rolling'] = df['pos_change'].rolling(lag, min_periods=1).sum()
    df.drop(columns=[f'{lag}m_ret'], inplace=True)

def generate_simple_features(df):
    df['open_change'] = df.open.pct_change()
    df['high_change'] = df.high.pct_change()
    df['low_change'] = df.low.pct_change()
    df['close_change'] = df.close.pct_change()
    df['volume_change'] = df.volume.pct_change()
    df.dropna(inplace=True)