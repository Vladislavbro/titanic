"""Песочница: MLP руками на разбиении 80/20, без hydra, wandb и cross_validate."""

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.sparse import issparse, spmatrix
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from titanic.data import prepare_data, save_submission
from titanic.train import make_cat_pipeline, make_num_pipeline

CONFIG = {
    "seed": 42,
    "test_size": 0.2,
    "data": {
        "train_path": "data/raw/train.csv",
        "test_path": "data/raw/test.csv",
        "submission_path": "submissions",
        "target_col": "Survived",
        "id_col": "PassengerId",
        "num_features": ["Age", "SibSp", "Parch", "Fare"],
        "cat_features": ["Sex", "Embarked", "Title", "FamilySize", "Pclass"],
    },
    "preprocessing": {
        "num": {"impute": "median", "add_flag": False, "scale": True},
        "cat": {"impute": "most_frequent", "encoding": "onehot", "random_state": 42},
    },
    "model": {
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "batch_size": 64,
        "epochs": 100,
    },
    "experiment": {"wandb": {"name": "sandbox_mlp"}},  # имя файла сабмишена
}


def build_preprocessor(cfg: DictConfig) -> ColumnTransformer:
    """Собирает препроцессинг числовых и категориальных фичей."""
    return ColumnTransformer(
        [
            (
                "num",
                make_num_pipeline(**cfg.preprocessing.num),
                list(cfg.data.num_features),
            ),
            (
                "cat",
                make_cat_pipeline(**cfg.preprocessing.cat),
                list(cfg.data.cat_features),
            ),
        ]
    )


def to_tensor(X: np.ndarray | spmatrix) -> torch.Tensor:
    """Переводит матрицу признаков в float32-тензор."""
    X = X.toarray() if issparse(X) else np.asarray(X)
    return torch.tensor(X, dtype=torch.float32)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Проводит одну эпоху обучения и возвращает средний лосс."""
    model.train()
    loss_sum = 0.0
    n_samples = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(xb)  # усредним сразу все в return
        n_samples += len(xb)
    return loss_sum / n_samples


def predict_proba(
    model: nn.Module, X: np.ndarray | spmatrix, batch_size: int, device: torch.device
) -> np.ndarray:
    """Считает вероятность класса 1 для каждого объекта."""
    model.eval()
    loader = DataLoader(TensorDataset(to_tensor(X)), batch_size=batch_size)
    batches = []
    with torch.no_grad():
        for (xb,) in loader:
            batches.append(torch.sigmoid(model(xb.to(device))))
    return torch.cat(batches).cpu().numpy().ravel()


def main() -> None:
    """Обучает MLP на 80/20, печатает метрики и пишет сабмишен."""
    cfg = OmegaConf.create(CONFIG)
    torch.manual_seed(cfg.seed)

    X_train, X_test, y_train, test_ids = prepare_data(cfg)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=cfg.test_size,
        stratify=y_train,
        random_state=cfg.seed,
    )

    preprocessor = build_preprocessor(cfg)
    X_tr = preprocessor.fit_transform(X_tr)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)
    print(f"train {X_tr.shape}, val {X_val.shape}, test {X_test.shape}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = nn.Sequential(
        nn.Linear(X_tr.shape[1], 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.model.lr, weight_decay=cfg.model.weight_decay
    )

    train_loader = DataLoader(
        TensorDataset(to_tensor(X_tr), to_tensor(y_tr).reshape(-1, 1)),
        batch_size=cfg.model.batch_size,
        shuffle=True,
    )

    for epoch in range(1, cfg.model.epochs + 1):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        proba = predict_proba(model, X_val, cfg.model.batch_size, device)
        print(
            f"эпоха {epoch:3d}  train loss {loss:.4f}"
            f"  val auc {roc_auc_score(y_val, proba):.4f}"
            f"  val logloss {log_loss(y_val, proba):.4f}"
        )

    proba = predict_proba(model, X_val, cfg.model.batch_size, device)
    print(f"val accuracy {accuracy_score(y_val, proba >= 0.5):.4f}")

    preds = (predict_proba(model, X_test, cfg.model.batch_size, device) >= 0.5).astype(
        int
    )
    save_submission(test_ids, preds, cfg)
    print(f"сабмишен: {cfg.data.submission_path}/{cfg.experiment.wandb.name}")


if __name__ == "__main__":
    main()
