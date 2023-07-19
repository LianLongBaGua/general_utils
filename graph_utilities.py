from datetime import datetime
from vnpy_ctastrategy.backtesting import BacktestingEngine, OptimizationSetting
import plotly.graph_objects as go

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def graph_result(x, y, result):

    z_dict = {}
    for param_str, target, statistics in result:
        param = eval(param_str)
        z_dict[(param[x], param[y])] = target

    z = []
    for x_value in x:
        z_buf = []
        for y_value in y:
            z_value = z_dict[(x_value, y_value)]
            z_buf.append(z_value)
        z.append(z_buf)   

    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
    fig.update_layout(
        title='RESULT', autosize = False,
        width=600, height=600,
        scene={
            'xaxis':{'title': x},
            'yaxis':{'title': y},
            'zaxis':{'title': Optimization_Setting.target_name},
        },
        margin={'l':65, 'r':50, 'b':65, 't':90}
    )
    fig.show()


def graph_two_ys(y_1: np.ndarray, label_1: str, y_2: np.ndarray, label_2: str):
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_ylabel(label_1, color=color)
    ax1.plot(y_1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  

    color = 'tab:blue'
    ax2.set_ylabel(label_2, color=color)  
    ax2.plot(y_2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  
    plt.show()
