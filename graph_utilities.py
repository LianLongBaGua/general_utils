from datetime import datetime
from vnpy_ctastrategy.backtesting import BacktestingEngine, OptimizationSetting
import plotly.graph_objects as go

import seaborn as sns
from usefultools.useful_tools import *
import pandas as pd


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
            'zaxis':{'title': optimization_setting.target_name},
        },
        margin={'l':65, 'r':50, 'b':65, 't':90}
    )
    fig.show()
