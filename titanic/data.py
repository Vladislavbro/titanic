from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from titanic.features import make_features

ROOT = Path(__file__).parents[1]


def load_data(cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    X_train = pd.read_csv(ROOT / cfg.data.train_path)
    X_test = pd.read_csv(ROOT / cfg.data.test_path)
    y_train = X_train.pop(cfg.data.target_col)
    return X_train, X_test, y_train


def prepare_data(
    cfg: DictConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Загружает данные и строит фичи."""
    X_train, X_test, y_train = load_data(cfg)
    train_ids = X_train[cfg.data.id_col]
    test_ids = X_test[cfg.data.id_col]
    X_train = make_features(X_train, cfg.data)
    X_test = make_features(X_test, cfg.data)
    return X_train, X_test, y_train, train_ids, test_ids


# PassengerId, y, proba.
def save_oof(
    cfg: DictConfig,
    train_ids: pd.Series,
    y_train: pd.Series,
    oof: np.ndarray,
) -> None:
    """Сохраняет OOF-вероятности на трейне."""
    path = ROOT / cfg.data.submission_path
    path.mkdir(parents=True, exist_ok=True)

    out = pd.DataFrame(
        {
            f"{cfg.data.id_col}": train_ids,
            "y": y_train,
            "proba": oof,
        }
    )
    out.to_csv(path / f"{cfg.experiment.wandb.name}_oof.csv", index=False)


def save_submission(
    test_ids: pd.Series,
    probs: np.ndarray,
    cfg: DictConfig,
    name: str | None = None,
) -> None:
    """Сохраняет вероятности и классы предсказаний на тесте."""
    path = ROOT / cfg.data.submission_path
    path.mkdir(parents=True, exist_ok=True)
    stem = name or cfg.experiment.wandb.name

    proba = pd.DataFrame({cfg.data.id_col: test_ids, "proba": probs})
    proba.to_csv(path / f"{stem}_test_proba.csv", index=False)

    submission = pd.DataFrame(
        {cfg.data.id_col: test_ids, cfg.data.target_col: (probs >= 0.5).astype(int)}
    )
    submission.to_csv(path / f"{stem}.csv", index=False)
