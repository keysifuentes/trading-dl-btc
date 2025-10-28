import numpy as np
import pandas as pd

def label_future_returns(df: pd.DataFrame, horizon=5, upper=0.01, lower=-0.01):
    c = df['Close']
    ret_f = c.shift(-horizon) / c - 1.0
    y = np.where(ret_f > upper, 1, np.where(ret_f < lower, -1, 0))
    y = pd.Series(y, index=df.index)
    return y.iloc[:-horizon], ret_f.iloc[:-horizon]
