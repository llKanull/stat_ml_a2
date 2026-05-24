from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from torch.utils.data import DataLoader, TensorDataset


# Logistic Regression (Simple model)
class LogisticModel:
    def __init__(self, C: float = 0.01) -> None:
        self.C = float(C)
        self._scaler: StandardScaler = StandardScaler()
        self._clf: LogisticRegression = LogisticRegression(
            C=self.C,
            
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticModel":
        X = self._scaler.fit_transform(np.asarray(X, dtype=float))
        self._clf.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._scaler.transform(np.asarray(X, dtype=float))
        return self._clf.predict(X)

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        X = self._scaler.transform(np.asarray(X, dtype=float))
        return self._clf.predict_proba(X)[:, 1]


# CatBoost (Middle complexity model)
class CatBoostModel:
    def __init__(
        self,
        depth: int = 9,
        learning_rate: float = 0.1,
        iterations: int = 200,
    ) -> None:
        self.depth = int(depth)
        self.learning_rate = float(learning_rate)
        self.iterations = int(iterations)
        self._clf = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CatBoostModel":
        self._clf = CatBoostClassifier(
            depth=self.depth,
            learning_rate=self.learning_rate,
            iterations=self.iterations,
            random_seed=42,
            verbose=0,
            allow_writing_files=False,
        )
        self._clf.fit(np.asarray(X, dtype=float), np.asarray(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict(np.asarray(X, dtype=float)).ravel()

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict_proba(np.asarray(X, dtype=float))[:, 1]


# FT-Transformer (Complex model)
class _FTTransformerNet:
    @staticmethod
    def build(
        n_features: int,
        n_blocks: int,
        d_token: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float,
    ):
        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()

                # Tokenizer 
                self.W = nn.Parameter(torch.empty(n_features, d_token))
                self.B = nn.Parameter(torch.zeros(n_features, d_token))
                nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
                self.cls = nn.Parameter(torch.empty(1, 1, d_token))
                nn.init.normal_(self.cls, std=0.02)

                # Transformer encoder 
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_token,
                    nhead=n_heads,
                    dim_feedforward=ffn_dim,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=n_blocks,
                    norm=nn.LayerNorm(d_token), enable_nested_tensor=False,
                )

                # Classification head
                self.head = nn.Sequential(
                    nn.LayerNorm(d_token),
                    nn.Linear(d_token, 1),
                )

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                tokens = x.unsqueeze(-1) * self.W + self.B  
                cls = self.cls.expand(x.size(0), -1, -1)    
                tokens = torch.cat([cls, tokens], dim=1)     
                encoded = self.encoder(tokens)               
                return self.head(encoded[:, 0, :])          

        return Net()


# FT-Transformer wrapper 
class FTTransformerModel:
    def __init__(
        self,
        n_blocks: int = 4,
        d_token: int = 32,       # was 64 — halves parameter count
        n_heads: int = 4,        # was 8
        ffn_dim: int = 128,      # was 256
        dropout: float = 0.1,
        lr: float = 3e-4,
        max_epochs: int = 15,    # was 50 — biggest speedup
        patience: int = 5,       # was 10
        batch_size: int = 512,
    ) -> None:
        self.n_blocks = int(n_blocks)
        self.d_token = int(d_token)
        self.n_heads = int(n_heads)
        self.ffn_dim = int(ffn_dim)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.batch_size = int(batch_size)
        self._scaler: StandardScaler = StandardScaler()
        self._net = None
        self._device = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FTTransformerModel":
        X = self._scaler.fit_transform(np.asarray(X, dtype=np.float32))
        y = np.asarray(y, dtype=np.float32)

        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        rng = np.random.default_rng(42)
        n_val = max(1, int(0.10 * len(X)))
        idx = rng.permutation(len(X))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]

        def _t(arr):
            return torch.tensor(arr, device=self._device)

        X_tr, y_tr = _t(X[tr_idx]), _t(y[tr_idx])
        X_val, y_val = _t(X[val_idx]), _t(y[val_idx])

        loader = DataLoader(
            TensorDataset(X_tr, y_tr),
            batch_size=min(self.batch_size, len(tr_idx)),
            shuffle=True,
        )

        self._net = _FTTransformerNet.build(
            n_features=X.shape[1],
            n_blocks=self.n_blocks,
            d_token=self.d_token,
            n_heads=self.n_heads,
            ffn_dim=self.ffn_dim,
            dropout=self.dropout,
        ).to(self._device)

        optimiser = torch.optim.AdamW(
            self._net.parameters(), lr=self.lr, weight_decay=1e-5
        )
        criterion = nn.BCEWithLogitsLoss()
        best_val_loss = float("inf")
        best_state: dict | None = None
        patience_counter = 0

        for _ in range(self.max_epochs):
            # Training
            self._net.train()
            for X_batch, y_batch in loader:
                optimiser.zero_grad()
                logits = self._net(X_batch).squeeze(-1)
                criterion(logits, y_batch).backward()
                optimiser.step()

            # Validation
            self._net.eval()
            with torch.no_grad():
                val_loss = criterion(
                    self._net(X_val).squeeze(-1), y_val
                ).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    k: v.cpu().clone()
                    for k, v in self._net.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)

        return self

    def _proba(self, X: np.ndarray) -> np.ndarray:
        X_t = torch.tensor(
            self._scaler.transform(np.asarray(X, dtype=np.float32)),
            device=self._device,
        )
        self._net.eval()
        with torch.no_grad():
            return torch.sigmoid(
                self._net(X_t).squeeze(-1)
            ).cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self._proba(X) >= 0.5).astype(int)

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        return self._proba(X)