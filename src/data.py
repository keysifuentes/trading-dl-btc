import pandas as pd

NUM_COLS = ["Open", "High", "Low", "Close", "Volume"]

def load_prices(csv_path: str) -> pd.DataFrame:
    # Leer CSV
    df = pd.read_csv(csv_path)

    # Validar y convertir Date
    if "Date" not in df.columns:
        raise ValueError("El CSV debe tener una columna 'Date'.")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Normalizar nombres (ej. ' close ' -> 'Close')
    df = df.rename(columns={c: c.strip().title() for c in df.columns})

    # Forzar columnas numéricas
    for col in NUM_COLS:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("—", "", regex=False)
                .str.replace(" ", "", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Orden y limpieza
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.dropna(subset=["Date"] + [c for c in NUM_COLS if c in df.columns])

    # Validación columnas
    missing = [c for c in NUM_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")

    return df
