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


def run_dir(cfg: DictConfig, name: str | None = None) -> Path:
    """Возвращает папку прогона в submissions, создавая её при необходимости."""
    path = ROOT / cfg.data.submission_path / (name or cfg.experiment.wandb.name)
    path.mkdir(parents=True, exist_ok=True)
    return path


# PassengerId, y, proba.
def save_oof(
    cfg: DictConfig,
    train_ids: pd.Series,
    y_train: pd.Series,
    oof: np.ndarray,
) -> None:
    """Сохраняет OOF-вероятности на трейне."""
    out = pd.DataFrame(
        {
            f"{cfg.data.id_col}": train_ids,
            "y": y_train,
            "proba": oof,
        }
    )
    out.to_csv(run_dir(cfg) / "oof.csv", index=False)


def save_submission(
    test_ids: pd.Series,
    probs: np.ndarray,
    cfg: DictConfig,
    name: str | None = None,
) -> None:
    """Сохраняет вероятности и классы предсказаний на тесте."""
    path = run_dir(cfg, name)

    proba = pd.DataFrame({cfg.data.id_col: test_ids, "proba": probs})
    proba.to_csv(path / "test_proba.csv", index=False)

    submission = pd.DataFrame(
        {cfg.data.id_col: test_ids, cfg.data.target_col: (probs >= 0.5).astype(int)}
    )
    submission.to_csv(path / f"{path.name}.csv", index=False)
