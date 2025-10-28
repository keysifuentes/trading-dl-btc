import os
import pandas as pd
import yfinance as yf

OUT_PATH = "data/prices.csv"
os.makedirs("data", exist_ok=True)

def main():
    # Fuerza 15+ años de histórico
    df = yf.download("BTC-USD", start="2010-01-01", interval="1d", auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError("No se pudo descargar BTC-USD desde Yahoo Finance. Revisa tu conexión o intenta de nuevo.")
    df = df.reset_index()
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna()
    df = df.sort_values("Date").reset_index(drop=True)
    df.to_csv(OUT_PATH, index=False)

    abs_path = os.path.abspath(OUT_PATH)
    print(f"Datos guardados en {abs_path} ({len(df)} filas). "
          f"Rango: {df['Date'].iloc[0].date()} → {df['Date'].iloc[-1].date()}")

if __name__ == "__main__":
    main()
