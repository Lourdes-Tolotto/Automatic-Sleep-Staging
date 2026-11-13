"""Arquitecturas de NB6 y NB12, sin alterar nombres de parámetros."""
import torch
from torch import nn

class SleepCNN(nn.Module):
    def __init__(self, n_channels=7, n_classes=5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(128, n_classes),
        )

    def encode(self, x):
        return self.features(x).squeeze(-1)

    def forward(self, x):
        return self.classifier(self.features(x))


class TemporalBiGRU(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, n_classes=5):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(0.30)
        self.classifier = nn.Linear(
            hidden_dim * 2,
            n_classes,
        )

    def forward(self, x):
        sequence_output, _ = self.gru(x)
        center = x.shape[1] // 2
        center_output = sequence_output[:, center, :]

        return self.classifier(
            self.dropout(center_output)
        )
