import os
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(path: str): os.makedirs(path, exist_ok=True)

def rolling_drawdown(equity: np.ndarray):
    peak = np.maximum.accumulate(equity)
    return equity / peak - 1.0

def max_drawdown(equity: np.ndarray):
    return float(np.min(rolling_drawdown(equity)))

def sharpe(returns: np.ndarray, risk_free_daily: float = 0.0):
    if returns.size == 0 or returns.std(ddof=1) == 0: return 0.0
    excess = returns - risk_free_daily
    return float(np.sqrt(252) * excess.mean() / (excess.std(ddof=1) + 1e-12))

def sortino(returns: np.ndarray, risk_free_daily: float = 0.0):
    if returns.size == 0: return 0.0
    downside = np.std(np.minimum(0, returns - risk_free_daily), ddof=1)
    if downside == 0: return 0.0
    return float(np.sqrt(252) * (returns.mean() - risk_free_daily) / (downside + 1e-12))

def calmar(equity: np.ndarray):
    if len(equity) < 2: return 0.0
    years = max(len(equity) / 252.0, 1e-9)
    cagr = (equity[-1] / equity[0]) ** (1 / years) - 1.0
    mdd = abs(max_drawdown(equity))
    return 0.0 if mdd == 0 else float(cagr / mdd)

def plot_equity_curve(dates, equity, out_path):
    plt.figure(figsize=(10,4))
    plt.plot(dates, equity)
    plt.title("Equity Curve")
    plt.xlabel("Date"); plt.ylabel("Equity")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
