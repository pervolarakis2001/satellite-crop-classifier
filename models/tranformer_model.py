import torch
import torch.nn as nn
import math
import json
from datetime import datetime

class SinPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dates_json_path: str, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        # Load dates from JSON
        with open(dates_json_path, "r") as f:
            raw_dates = json.load(f)

        # Convert dates to number of days since the first date
        dates = [str(d) for d in raw_dates]
        start = datetime.strptime(dates[0], "%Y%m%d")
        self.day_deltas = torch.tensor(
            [(datetime.strptime(d, "%Y%m%d") - start).days for d in dates],
            dtype=torch.float32
        )

    def forward(self, x, acquisition_times):
        B, T, D = x.shape
        pe = torch.zeros_like(x)

        div_term = torch.exp(
            torch.arange(0, D, 2, device=x.device) * (-math.log(10000.0) / D)
        )

        times = acquisition_times.to(x.device).unsqueeze(-1)
        pe[..., 0::2] = torch.sin(times * div_term)
        pe[..., 1::2] = torch.cos(times * div_term)

        return self.dropout(x + pe)


class TransformerWithCLS(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dates_json_path: str,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, input_dim))

        self.pos_encoder = SinPositionalEncoding(
            d_model=input_dim,
            dates_json_path=dates_json_path,
            dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        #MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, acquisition_times):
        B = x.size(0)

        x = self.pos_encoder(x, acquisition_times)
        cls_token = self.cls_token.expand(B, 1, self.input_dim)
        x = torch.cat([cls_token, x], dim=1)
        x = self.transformer(x)
        cls_out = x[:, 0, :]
        logits = self.classifier(cls_out)
        return logits  #OUPUT [B, num_classes]