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


def load_and_preprocess_data(file_path, train_sheet, test_sheet, target_column):
    
    train_df = pd.read_excel(file_path, sheet_name=train_sheet)
    test_df = pd.read_excel(file_path, sheet_name=test_sheet)

    
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    
    combined = pd.concat([X_train, X_test], axis=0)
    combined_encoded = pd.get_dummies(combined)

    
    X_train_encoded = combined_encoded.iloc[:X_train.shape[0], :]
    X_test_encoded = combined_encoded.iloc[X_train.shape[0]:, :]

    
    return (
        torch.tensor(X_train_encoded.values, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.float32),
        torch.tensor(X_test_encoded.values, dtype=torch.float32),
        torch.tensor(y_test.values, dtype=torch.float32)
    )



def train_transformer_mlp_model(X_train, y_train, input_dim, epochs=100, batch_size=32, learning_rate=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerMLP(input_dim, hidden_dim=100, transformer_layers=5, num_heads=5, ff_dim=256, output_dim=1)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    dataset = TensorDataset(X_train, y_train.unsqueeze(1))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    best_model = None
    best_score = -np.inf
    train_predictions = []
    model.train()
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        epoch_predictions = []
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_predictions.extend(y_pred.cpu().detach().numpy().flatten())
        r2 = r2_score(y_train.numpy(), epoch_predictions)
        if r2 > best_score:
            best_score = r2
            best_model = model.state_dict()
        train_predictions = epoch_predictions
    model.load_state_dict(best_model)
    return model, train_predictions

