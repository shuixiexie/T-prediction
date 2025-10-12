import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from tqdm import tqdm

class TransformerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, transformer_layers, num_heads, ff_dim, output_dim):
        super(TransformerMLP, self).__init__()
        if input_dim % num_heads != 0:
            for i in range(num_heads, 0, -1):
                if input_dim % i == 0:
                    num_heads = i
                    break
            print(f"Adjusted num_heads to {num_heads} for compatibility with input_dim {input_dim}")
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=transformer_layers)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = x.mean(dim=1)
        return self.fc(x)





