import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

from src.models import MLP, CNN1D

class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len=20, for_sequence=False):
        self.for_sequence = for_sequence
        self.seq_len = seq_len
        if for_sequence:
            seqX, sy = [], []
            L = len(X)
            if L > seq_len:
                for i in range(seq_len, L):
                    seqX.append(X[i-seq_len:i])
                    sy.append(y[i])
            self.X = np.array(seqX) if len(seqX) else np.empty((0, seq_len, X.shape[1]), dtype=np.float32)
            self.y = np.array(sy) if len(sy) else np.empty((0,), dtype=np.int64)
        else:
            self.X = X
            self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

def class_weights_from(y_tr_raw):
    """
    y_tr_raw en {-1,0,1}. Devuelve pesos en orden de índices mapeados [0,1,2] = [-1,0,1].
    """
    classes, counts = np.unique(y_tr_raw, return_counts=True)
    total = counts.sum()
    w = {int(c): total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    # índice 0 corresponde a la clase -1; índice 1 -> 0; índice 2 -> 1
    return [w.get(-1, 1.0), w.get(0, 1.0), w.get(1, 1.0)]

def train_one(model, train_loader, val_loader, class_weights, epochs=30, lr=1e-3, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
    best_f1, best_state = -1.0, None

    for ep in range(1, epochs+1):
        # Train
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        # Eval
        def _eval(loader):
            ys, yh = [], []
            model.eval()
            with torch.no_grad():
                for xb, yb in loader:
                    xb = xb.to(device)
                    pred = model(xb).argmax(dim=1).cpu().numpy()
                    ys.append(yb.numpy()); yh.append(pred)
            if not ys:
                return -1.0  # no hay datos
            y_true = np.concatenate(ys); y_pred = np.concatenate(yh)
            return f1_score(y_true, y_pred, average='macro')

        f1_tr = _eval(train_loader)
        f1_va = _eval(val_loader)
        mlflow.log_metrics({"f1_train": f1_tr, "f1_val": f1_va}, step=ep)

        if f1_va > best_f1:
            best_f1 = f1_va
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_f1

def run_experiment(X_tr, y_tr_raw, X_va, y_va_raw, arch='mlp',
                   seq_len=20, batch_size=64, epochs=25, lr=1e-3, device='cpu'):
    # Pesos con etiquetas originales (-1,0,1)
    class_weights = class_weights_from(y_tr_raw)

    # Mapear etiquetas a 0,1,2 para CrossEntropy
    y_tr = (y_tr_raw + 1).astype(int)
    y_va = (y_va_raw + 1).astype(int)

    if arch == 'mlp':
        ds_tr = SeqDataset(X_tr, y_tr, for_sequence=False)
        ds_va = SeqDataset(X_va, y_va, for_sequence=False)
        model = MLP(in_dim=X_tr.shape[1])
    elif arch == 'cnn':
        if len(X_tr) <= seq_len or len(X_va) <= seq_len:
            raise ValueError(f"Datos insuficientes para CNN con seq_len={seq_len}. "
                             f"len(X_tr)={len(X_tr)}, len(X_va)={len(X_va)}")
        ds_tr = SeqDataset(X_tr, y_tr, seq_len=seq_len, for_sequence=True)
        ds_va = SeqDataset(X_va, y_va, seq_len=seq_len, for_sequence=True)
        if len(ds_tr) == 0 or len(ds_va) == 0:
            raise ValueError("SeqDataset está vacío; no se pueden crear secuencias para CNN.")
        model = CNN1D(in_channels=X_tr.shape[1])
    else:
        raise ValueError("arch debe ser 'mlp' o 'cnn'")

    tr_loader = DataLoader(ds_tr, batch_size=min(batch_size, max(1, len(ds_tr))), shuffle=True)
    va_loader = DataLoader(ds_va, batch_size=min(batch_size, max(1, len(ds_va))), shuffle=False)

    mlflow.log_params({
        "arch": arch, "seq_len": seq_len, "batch_size": batch_size,
        "epochs": epochs, "lr": lr
    })

    model, best_f1 = train_one(model, tr_loader, va_loader, class_weights, epochs, lr, device)
    mlflow.pytorch.log_model(model, artifact_path=f"model_{arch}")
    return model, best_f1
