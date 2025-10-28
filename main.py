import os
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler

from src.data import load_prices
from src.features import engineer_features
from src.labels import label_future_returns
from src.train import run_experiment
from src.metrics import confusion_counts, accuracy
from src.drift import ks_drift_table, timeline_feature_plot
from src.utils import ensure_dir, plot_equity_curve
from src.backtest import backtest, logits_to_signals

DATA_CSV = "data/prices.csv"
OUT_DIR = "outputs"
MLFLOW_EXP = "trading-dl-mlp-cnn"

def main():
    ensure_dir(OUT_DIR)

    # 1) Carga de datos
    prices = load_prices(DATA_CSV)
    print(f"[INFO] Filas crudas en prices: {len(prices)} | "
          f"rango {prices['Date'].iloc[0].date()} → {prices['Date'].iloc[-1].date()}")

    # 2) Features (auto short_mode si hay pocos datos)
    short_mode = len(prices) < 300
    feats = engineer_features(prices, short_mode=short_mode)
    print(f"[INFO] Filas después de features: {len(feats)}")

    # 3) Alinea precios con las filas válidas de features
    aligned = prices.loc[feats.index].reset_index(drop=True)
    feats = feats.reset_index(drop=True)

    # 4) Labels (sin look-ahead)
    horizon = 3 if short_mode else 5
    upper, lower = (0.02, -0.02) if short_mode else (0.01, -0.01)
    y, ret_future = label_future_returns(aligned, horizon=horizon, upper=upper, lower=lower)

    # 5) Asegurar X e y del mismo largo
    min_len = min(len(feats), len(y))
    feats = feats.iloc[:min_len].reset_index(drop=True)
    y = y.iloc[:min_len].reset_index(drop=True)
    aligned = aligned.iloc[:min_len].reset_index(drop=True)

    X = feats
    y = y.values

    print(f"[INFO] Filas finales para modelado: {len(X)} (short_mode={short_mode}, horizon={horizon})")

    # Si sigue siendo poquísimo, avisar y salir con instrucción clara
    if len(X) < 200:
        raise RuntimeError(
            f"Hay muy pocos datos tras el preprocesado (n={len(X)}). "
            f"Revisa que 'python data_download.py' bajó miles de filas, o intenta short_mode reduciendo aún más ventanas."
        )

    # 6) Splits 60/20/20
    n = len(X)
    n_train = int(0.6 * n)
    n_test  = int(0.2 * n)

    idx_tr = np.arange(0, n_train)
    idx_te = np.arange(n_train, n_train + n_test)
    idx_va = np.arange(n_train + n_test, n)

    X_tr, X_te, X_va = X.iloc[idx_tr], X.iloc[idx_te], X.iloc[idx_va]
    y_tr, y_te, y_va = y[idx_tr], y[idx_te], y[idx_va]

    # 7) Escalado (fit solo en TRAIN)
    scaler = RobustScaler().fit(X_tr.values)
    Xt_tr = scaler.transform(X_tr.values)
    Xt_te = scaler.transform(X_te.values)
    Xt_va = scaler.transform(X_va.values)

    # 8) Dispositivo y experimento MLflow
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mlflow.set_experiment(MLFLOW_EXP)

    # 9) Determinar seq_len (CNN) dependiendo del tamaño real
    def seq_len_dyn(max_len=20):
        caps = [len(X_tr)-1, len(X_va)-1, len(X_te)-1]
        caps = [c for c in caps if c is not None and c > 1]
        return None if not caps else max(2, min(max_len, *caps))

    seq_len = seq_len_dyn(20)
    can_run_cnn = (not short_mode) and seq_len is not None and (len(X_tr) > seq_len) and (len(X_va) > seq_len)
    archs = ["mlp"] + (["cnn"] if can_run_cnn else [])

    if not can_run_cnn:
        print(f"[INFO] CNN saltada o en short_mode={short_mode}; usaremos solo MLP (seq_len sugerido={seq_len}).")

    # 10) Entrenar y comparar
    results = []
    for arch in archs:
        with mlflow.start_run(run_name=f"{arch}"):
            model, f1_val = run_experiment(
                Xt_tr, y_tr, Xt_va, y_va,
                arch=arch, seq_len=(seq_len or 10), batch_size=64, epochs=25, lr=1e-3, device=device
            )

            if arch == "mlp":
                logits_va = model(torch.tensor(Xt_va, dtype=torch.float32).to(device)).detach().cpu().numpy()
                y_va_eff = y_va
                prices_va = aligned.iloc[idx_va].reset_index(drop=True)
            else:
                s = seq_len or 10
                seqs = [Xt_va[i-s:i] for i in range(s, len(Xt_va))]
                logits_va = model(torch.tensor(np.array(seqs), dtype=torch.float32).to(device)).detach().cpu().numpy()
                y_va_eff = y_va[s:]
                prices_va = aligned.iloc[idx_va].iloc[s:].reset_index(drop=True)

            # Backtest validación
            signals_va, _ = logits_to_signals(logits_va, thr_long=0.2, thr_short=0.2)
            bt = backtest(prices_va, signals_va, commission=0.00125, borrow_rate_annual=0.0025)

            y_pred_va = np.argmax(logits_va, axis=1) - 1
            acc_va = accuracy(y_va_eff, y_pred_va)

            mlflow.log_metrics({
                "f1_val": f1_val, "acc_val": acc_va,
                "bt_sharpe_val": bt["sharpe"], "bt_sortino_val": bt["sortino"],
                "bt_calmar_val": bt["calmar"], "bt_mdd_val": bt["mdd"],
                "bt_trades_val": bt["trades"], "bt_winrate_val": bt["winrate"]
            })

            cm_path = os.path.join(OUT_DIR, f"confusion_{arch}_val.csv")
            confusion_counts(y_va_eff, y_pred_va).to_csv(cm_path)

            results.append({"arch": arch, "f1_val": f1_val, "acc_val": acc_va, "calmar_val": bt["calmar"]})

    if not results:
        raise RuntimeError("No se pudo entrenar ningún modelo. Verifica tamaño de datos y vuelve a ejecutar.")

    best = max(results, key=lambda r: r["calmar_val"])
    print("Resultados VALIDATION:", results)
    print("Mejor modelo:", best)

    # 11) Drift
    ks_test = ks_drift_table(pd.DataFrame(Xt_tr, columns=X.columns), pd.DataFrame(Xt_te, columns=X.columns), "test")
    ks_val  = ks_drift_table(pd.DataFrame(Xt_tr, columns=X.columns), pd.DataFrame(Xt_va, columns=X.columns), "val")
    ks_test.to_csv(os.path.join(OUT_DIR, "drift_ks_test.csv"), index=False)
    ks_val.to_csv(os.path.join(OUT_DIR, "drift_ks_val.csv"), index=False)
    timeline_feature_plot(X.iloc[min(idx_tr):max(idx_va)+1], aligned.iloc[min(idx_tr):max(idx_va)+1]['Date'],
                          os.path.join(OUT_DIR, "timeline_features"), limit=25)

    # 12) Backtest test con mejor modelo
    arch = best["arch"]
    with mlflow.start_run(run_name=f"{arch}_final_test"):
        model, _ = run_experiment(Xt_tr, y_tr, Xt_va, y_va, arch=arch, seq_len=(seq_len or 10), batch_size=64, epochs=25, lr=1e-3, device=device)

        if arch == "mlp":
            logits_te = model(torch.tensor(Xt_te, dtype=torch.float32).to(device)).detach().cpu().numpy()
            prices_te = aligned.iloc[idx_te].reset_index(drop=True)
            y_te_eff = y_te
        else:
            s = seq_len or 10
            seqs = [Xt_te[i-s:i] for i in range(s, len(Xt_te))]
            logits_te = model(torch.tensor(np.array(seqs), dtype=torch.float32).to(device)).detach().cpu().numpy()
            prices_te = aligned.iloc[idx_te].iloc[s:].reset_index(drop=True)
            y_te_eff = y_te[s:]

        signals_te, _ = logits_to_signals(logits_te, thr_long=0.2, thr_short=0.2)
        bt_te = backtest(prices_te, signals_te, commission=0.00125, borrow_rate_annual=0.0025)

        eq_path = os.path.join(OUT_DIR, f"equity_{arch}_test.png")
        plot_equity_curve(prices_te['Date'].values, bt_te["equity"], eq_path)
        mlflow.log_artifact(eq_path)

        y_pred_te = np.argmax(logits_te, axis=1) - 1
        cm_path = os.path.join(OUT_DIR, f"confusion_{arch}_test.csv")
        confusion_counts(y_te_eff, y_pred_te).to_csv(cm_path)
        mlflow.log_artifact(cm_path)

        print(f"[TEST] {arch}  Sharpe={bt_te['sharpe']:.3f}  Sortino={bt_te['sortino']:.3f}  "
              f"Calmar={bt_te['calmar']:.3f}  MDD={bt_te['mdd']:.3f}")
        print(f"[TEST] Trades={bt_te['trades']}  WinRate={bt_te['winrate']:.2%}")

if __name__ == "__main__":
    main()
