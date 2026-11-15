from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

ROOT = Path(__file__).parents[1]


def load_data(cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    X_train = pd.read_csv(cfg.data.train_path)
    X_test = pd.read_csv(cfg.data.test_path)
    y_train = X_train.pop(cfg.data.target_col)
    return X_train, X_test, y_train


def save_submission(
    test_ids: pd.Series,
    preds: np.ndarray,
    cfg: DictConfig,
    name: str = "submission",
) -> None:
    path = ROOT / cfg.data.submission_path
    path.mkdir(parents=True, exist_ok=True)

    submission = pd.DataFrame(
        {cfg.data.id_col: test_ids, cfg.data.target_col: preds.astype(int)}
    )
    submission.to_csv(path / f"{name}.csv", index=False)
