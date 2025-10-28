
import os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
from src.utils import ensure_dir

def ks_drift_table(train_feats: pd.DataFrame, other_feats: pd.DataFrame, name_other="split"):
    rows = []
    for col in train_feats.columns:
        a = train_feats[col].dropna().values
        b = other_feats[col].dropna().values
        if len(a) > 30 and len(b) > 30:
            stat, p = ks_2samp(a, b)
        else:
            stat, p = np.nan, np.nan
        rows.append({"feature": col, "ks_stat": stat, "p_value": p,
                     "drift": (p < 0.05) if not np.isnan(p) else False,
                     "against": name_other})
    return pd.DataFrame(rows).sort_values("p_value")

def timeline_feature_plot(feats: pd.DataFrame, dates: pd.Series, out_dir: str, limit=25):
    ensure_dir(out_dir)
    for col in list(feats.columns)[:limit]:
        plt.figure(figsize=(10,3))
        plt.plot(dates, feats[col])
        plt.title(f"Feature over time: {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{col}.png"), dpi=120)
        plt.close()
