import numpy as np
import pandas as pd
from src.utils import sharpe, sortino, calmar, max_drawdown

def logits_to_signals(logits: np.ndarray, thr_long=0.2, thr_short=0.2):
    exps = np.exp(logits - logits.max(axis=1, keepdims=True))
    prob = exps / exps.sum(axis=1, keepdims=True)
    p_short, p_hold, p_long = prob[:, 0], prob[:, 1], prob[:, 2]
    sig = np.where((p_long - p_short) > thr_long, 1,
          np.where((p_short - p_long) > thr_short, -1, 0))
    return sig, prob

def backtest(prices: pd.DataFrame, signals: np.ndarray,
             commission=0.00125, borrow_rate_annual=0.0025):
    # Defensas por tamaño
    if len(prices) < 2 or len(signals) < 2:
        equity = np.array([1.0])
        return {
            "equity": equity,
            "returns": np.array([]),
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "mdd": 0.0,
            "trades": 0,
            "winrate": 0.0
        }

    close = prices['Close'].values
    ret_next = np.diff(close) / close[:-1]               # r_{t+1}
    sig_t = signals[:-1]                                  # s_t
    L = min(len(ret_next), len(sig_t))
    ret_next = ret_next[:L]
    sig_t = sig_t[:L]

    trades = np.diff(sig_t, prepend=0) != 0              # cambio de posición
    trade_cost = trades * commission                      # costo por cambio (simple)

    # Costo por short (borrow)
    borrow_daily = borrow_rate_annual / 252.0
    borrow_cost = (sig_t < 0).astype(float) * borrow_daily

    strat_ret = sig_t * ret_next - trade_cost - borrow_cost
    equity = np.concatenate([[1.0], (1 + strat_ret).cumprod()])

    # Métricas robustas en series cortas
    sharpe_val = sharpe(strat_ret) if strat_ret.size else 0.0
    sortino_val = sortino(strat_ret) if strat_ret.size else 0.0
    calmar_val = calmar(equity) if equity.size > 1 else 0.0
    mdd_val = max_drawdown(equity) if equity.size > 1 else 0.0
    winrate_val = float((strat_ret > 0).mean()) if strat_ret.size else 0.0

    return {
        "equity": equity,
        "returns": strat_ret,
        "sharpe": sharpe_val,
        "sortino": sortino_val,
        "calmar": calmar_val,
        "mdd": mdd_val,
        "trades": int(trades.sum()),
        "winrate": winrate_val
    }
