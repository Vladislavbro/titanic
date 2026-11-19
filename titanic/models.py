import numpy as np
import torch
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}

OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}


class FocalLoss(nn.Module):
    """Считает focal loss по логитам для бинарной классификации."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * bce * (1 - p_t) ** self.gamma).mean()


class MLPClassifier(ClassifierMixin, BaseEstimator):
    """Обучает MLP на табличных данных через sklearn-интерфейс."""

    def __init__(  # noqa: PLR0913
        self,
        hidden_sizes: tuple[int, ...] = (128,),
        activation: str = "relu",
        dropout: float = 0.0,
        batchnorm: bool = False,
        optimizer: str = "adamw",
        scheduler: str = "none",
        loss: str = "bce",
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        batch_size: int = 64,
        epochs: int = 100,
        device: str | None = None,
        random_state: int = 42,
    ) -> None:
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.random_state = random_state

    def _build_module(self, n_features: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        prev = n_features
        for size in self.hidden_sizes:
            layers.append(nn.Linear(prev, size))
            if self.batchnorm:
                layers.append(nn.BatchNorm1d(size))
            layers.append(ACTIVATIONS[self.activation]())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            prev = size
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def _make_criterion(self) -> nn.Module:
        if self.loss == "focal":
            return FocalLoss()
        return nn.BCEWithLogitsLoss()

    def _make_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        if self.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs
            )
        return None

    def _make_loader(self, X: torch.Tensor, y: torch.Tensor | None) -> DataLoader:
        dataset = TensorDataset(X) if y is None else TensorDataset(X, y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=y is not None)

    @staticmethod
    def _pick_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        X = X.toarray() if issparse(X) else np.asarray(X)
        return torch.tensor(X, dtype=torch.float32)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPClassifier":
        torch.manual_seed(self.random_state)

        self.classes_ = np.unique(np.asarray(y))
        self.n_features_in_ = X.shape[1]
        self.device_ = self.device or self._pick_device()
        self.module_ = self._build_module(self.n_features_in_).to(self.device_)

        X_tensor = self._to_tensor(X)
        y_tensor = torch.tensor(np.asarray(y, dtype=np.float32).reshape(-1, 1))
        loader = self._make_loader(X_tensor, y_tensor)

        criterion = self._make_criterion()
        optimizer = OPTIMIZERS[self.optimizer](
            self.module_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = self._make_scheduler(optimizer)

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
            for (xb,) in self._make_loader(self._to_tensor(X), None):
                batches.append(torch.sigmoid(self.module_(xb.to(self.device_))))
        proba = torch.cat(batches).cpu().numpy().ravel()
        return np.column_stack([1 - proba, proba])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
