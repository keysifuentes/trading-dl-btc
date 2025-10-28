import numpy as np
import pandas as pd

def confusion_counts(y_true, y_pred, labels=(-1, 0, 1)):
    df = pd.crosstab(pd.Series(y_true, name="true"),
                     pd.Series(y_pred, name="pred"),
                     dropna=False)
    return df.reindex(index=labels, columns=labels, fill_value=0)

def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())
