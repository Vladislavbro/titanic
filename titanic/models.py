import numpy as np
import torch
from scipy.sparse import issparse, spmatrix
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MLPClassifier(ClassifierMixin, BaseEstimator):
    """Обучает MLP с одним скрытым слоем на табличных данных."""

    def __init__(  # noqa: PLR0913
        self,
        hidden_size: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        batch_size: int = 64,
        epochs: int = 10,
        random_state: int = 42,
    ) -> None:
        self.hidden_size = hidden_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state

    def _pick_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _to_tensor(self, X: np.ndarray | spmatrix) -> torch.Tensor:
        X = X.toarray() if issparse(X) else np.asarray(X)
        return torch.tensor(X, dtype=torch.float32)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPClassifier":
        torch.manual_seed(self.random_state)
        self.classes_ = np.unique(np.asarray(y))
        self.n_features_in_ = X.shape[1]
        self.device_ = self._pick_device()

        X_tensor = self._to_tensor(X)
        y_tensor = torch.tensor(np.asarray(y, dtype=np.float32).reshape(-1, 1))
        loader = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.module_ = nn.Sequential(
            nn.Linear(self.n_features_in_, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        ).to(self.device_)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            self.module_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self.module_.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device_), yb.to(self.device_)
                optimizer.zero_grad()
                loss = criterion(self.module_(xb), yb)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.module_.eval()
        batches = []
        with torch.no_grad():
            loader = DataLoader(
                TensorDataset(self._to_tensor(X)), batch_size=self.batch_size
            )
            for (xb,) in loader:
                batches.append(torch.sigmoid(self.module_(xb.to(self.device_))))
        proba = torch.cat(batches).cpu().numpy().ravel()
        return np.column_stack([1 - proba, proba])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
