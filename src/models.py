import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(128, 64), out_dim=3, p=0.2):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(p)]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # (B, feat)
        return self.net(x)

class CNN1D(nn.Module):
    def __init__(self, in_channels, out_dim=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):  # (B, seq_len, feat)
        x = x.transpose(1, 2)
        z = self.conv(x).squeeze(-1)
        return self.fc(z)
