"""Прогон CatBoost на всём трейне: число деревьев подбирается early stopping по фолдам."""

import hydra
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score

from titanic.data import prepare_data, save_submission
from titanic.train import build_pipeline

EARLY_STOPPING_ROUNDS = 100


def fit_folds(cfg: DictConfig, X: pd.DataFrame, y: pd.Series) -> tuple[int, float]:
    """Обучает модель по фолдам с early stopping и возвращает число деревьев и средний скор."""
    cv = instantiate(cfg.validation.cv)
    iterations, scores = [], []
    for train_idx, val_idx in cv.split(X, y):
        preprocessor = build_pipeline(cfg).named_steps["preprocess"]
        X_tr = preprocessor.fit_transform(X.iloc[train_idx], y.iloc[train_idx])
        X_val = preprocessor.transform(X.iloc[val_idx])

        model = instantiate(cfg.model, eval_metric="Logloss")
        model.fit(
            X_tr,
            y.iloc[train_idx],
            eval_set=(X_val, y.iloc[val_idx]),
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        )
        iterations.append(model.get_best_iteration())
        scores.append(accuracy_score(y.iloc[val_idx], model.predict(X_val)))

    return int(np.mean(iterations)), float(np.mean(scores))


@hydra.main(
    config_path="../configs", config_name="catboost_earlystop", version_base=None
)
def main(cfg: DictConfig) -> None:
    """Учит CatBoost на всём трейне и пишет сабмит."""
    X_train, X_test, y_train, _, test_ids = prepare_data(cfg)

    iterations, score = fit_folds(cfg, X_train, y_train)
    print(f"деревьев {iterations}, точность по фолдам {score:.4f}")

    preprocessor = build_pipeline(cfg).named_steps["preprocess"]
    preprocessor.fit(X_train, y_train)
    model = instantiate(cfg.model, iterations=iterations)
    model.fit(preprocessor.transform(X_train), y_train)
    probs = model.predict_proba(preprocessor.transform(X_test))[:, 1]

    save_submission(test_ids, probs, cfg, name="catboost_earlystop")
    print("сабмишен: submissions/catboost_earlystop.csv")


if __name__ == "__main__":
    main()
