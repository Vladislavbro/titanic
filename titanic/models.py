import numpy as np
import torch
from scipy.sparse import issparse, spmatrix
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

OPTIMIZERS = {
    "adamw": torch.optim.AdamW,
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
}

ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}


# Наследуем sklearn-классы, чтобы модель встала в Pipeline и cross_validate
class MLPClassifier(ClassifierMixin, BaseEstimator):
    """Обучает MLP с одним скрытым слоем на табличных данных."""

    def __init__(  # noqa: PLR0913
        self,
        hidden_sizes: tuple[int, ...] = (128,),
        activation: str = "relu",
        dropout: float = 0.0,
        batchnorm: bool = False,
        optimizer: str = "adamw",
        scheduler: str = "none",
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        batch_size: int = 64,
        epochs: int = 10,
        device: str | None = None,
        random_state: int = 42,
    ) -> None:
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.random_state = random_state

    def _pick_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
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

        layers: list[nn.Module] = []
        in_size = self.n_features_in_
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))

            if self.batchnorm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(ACTIVATIONS[self.activation]())

            if self.dropout > 0:
                layers.append(nn.Dropout(p=self.dropout))
            in_size = hidden_size

        self.module_ = nn.Sequential(*layers, nn.Linear(in_size, 1)).to(self.device_)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = OPTIMIZERS[self.optimizer](
            self.module_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = None
        if self.scheduler == "cosine":  # снижает планово lr
            scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-5)

        self.module_.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device_), yb.to(self.device_)
                optimizer.zero_grad()
                loss = criterion(self.module_(xb), yb)
                loss.backward()
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

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
