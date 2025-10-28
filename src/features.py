import numpy as np
import pandas as pd

def rsi(series: pd.Series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = up / (down + 1e-12)
    return 100 - (100 / (1 + rs))

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def macd(close, fast=12, slow=26, signal=9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(close, window=20, num_std=2.0, minp=None):
    if minp is None: minp = max(2, int(window*0.5))
    ma = close.rolling(window, min_periods=minp).mean()
    std = close.rolling(window, min_periods=minp).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return ma, upper, lower, std

def atr_proxy(high, low, close, window=14, minp=None):
    if minp is None: minp = max(2, int(window*0.5))
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=minp).mean()

def zscore(series, window=20, minp=None):
    if minp is None: minp = max(2, int(window*0.5))
    mean = series.rolling(window, min_periods=minp).mean()
    std = series.rolling(window, min_periods=minp).std()
    return (series - mean) / (std + 1e-12)

def engineer_features(df: pd.DataFrame, short_mode=False) -> pd.DataFrame:
    """
    Genera >20 features con min_periods flexibles para no perder demasiadas filas.
    Luego rellena NaNs (bfill/ffill) para evitar drop masivo.
    """
    c = df['Close']; h = df['High']; l = df['Low']; v = df['Volume']
    out = pd.DataFrame(index=df.index)

    # Ventanas base (si el dataset fuera pequeño)
    if short_mode or len(df) < 300:
        w10, w14, w20, w26 = 7, 10, 14, 20
    else:
        w10, w14, w20, w26 = 10, 14, 20, 26

    # Momentum / returns
    out['ret_1']   = c.pct_change(1)
    out['ret_2']   = c.pct_change(2)
    out['ret_5']   = c.pct_change(5)
    out['ret_10']  = c.pct_change(10)
    out['mom_10']  = c / c.shift(w10) - 1
    out['mom_20']  = c / c.shift(max(2, w20)) - 1
    out['roc_5']   = c.pct_change(5)

    # Volatilidad
    out['vol_10']  = c.pct_change().rolling(w10, min_periods=max(2, int(w10*0.5))).std()
    out['vol_20']  = c.pct_change().rolling(w20, min_periods=max(2, int(w20*0.5))).std()
    ma, up, lo, bb_std = bollinger(c, w20, 2)
    out['bb_bw']   = (up - lo) / (ma + 1e-12)
    out['bb_pos']  = (c - ma) / (bb_std + 1e-12)
    out['atr14']   = atr_proxy(h, l, c, w14)

    # Tendencia
    out['sma_10']  = c.rolling(w10, min_periods=max(2, int(w10*0.5))).mean()
    out['sma_20']  = c.rolling(w20, min_periods=max(2, int(w20*0.5))).mean()
    out['ema_12']  = ema(c, 12 if not short_mode else max(6, int(w10*0.8)))
    out['ema_26']  = ema(c, w26)
    out['sma_slope_10'] = out['sma_10'].diff()
    out['ema_slope_12'] = out['ema_12'].diff()

    # MACD / RSI
    macd_line, signal, hist = macd(c, fast=12 if not short_mode else 8, slow=w26, signal=9)
    out['macd']     = macd_line
    out['macd_sig'] = signal
    out['macd_hist']= hist
    out['rsi_14']   = rsi(c, w14)

    # Volumen
    out['vol_z20']  = zscore(v, w20)
    out['vroc_10']  = v.pct_change(w10)

    # Sanitizar
    out = out.replace([np.inf, -np.inf], np.nan)
    # En lugar de drop masivo, rellenamos hacia atrás y adelante
    out = out.bfill().ffill()

    # Si aún quedaran NaNs puntuales (columnas planas al principio), los ponemos en 0
    out = out.fillna(0.0)

    return out

