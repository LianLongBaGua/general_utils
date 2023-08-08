from sklearn.preprocessing import StandardScaler
from pandas_ta import log_return
import pandas as pd
import numpy as np


def prepare_desired_pos(df, lag=50, multiplier=10):
    df = df.copy()
    scaler = StandardScaler()
    df[f'{lag}m_ret'] = scaler.fit_transform(log_return(df.close, length=lag, offset=-lag).values.reshape(-1,1))
    df.dropna(inplace=True)
    df['desired_pos_change'] = (df[f'{lag}m_ret'] * multiplier).apply(int)
    df['pos_change_signal'] = pd.qcut(df['desired_pos_change'], 5, ['strong sell', 'sell', 'meh', 'buy', 'strong buy'])
    df['desired_pos_rolling'] = df['desired_pos_change'].rolling(lag, min_periods=1).sum().apply(int)
    df['net_pos_signal'] = np.where(df['desired_pos_rolling'] > 0, 'long hold', 'short hold')
    df.drop(columns=[f'{lag}m_ret'], inplace=True)

    return df
