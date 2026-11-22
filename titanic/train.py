from typing import cast

import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    TargetEncoder,
)

import wandb


def make_num_pipeline(impute: str, add_flag: bool, scale: bool) -> Pipeline:
    """Собирает препроцессинг числовых фичей: импьютер и скейлер."""
    # Заполнение пропусков
    imputer = SimpleImputer(
        strategy=impute,
        add_indicator=add_flag,
    )
    steps = [("impute", imputer)]
    steps: list[tuple[str, TransformerMixin]] = [("impute", imputer)]

    # Масштабирование
    if scale:
        steps.append(("scale", StandardScaler()))

    return Pipeline(steps)


def make_cat_pipeline(impute: str, encoding: str, random_state: int) -> Pipeline:
    """Собирает препроцессинг категориальных фичей: импьютер и энкодер."""
    # Заполнение пропусков
    if impute == "most_frequent":
        imputer = SimpleImputer(strategy="most_frequent")
    elif impute == "missing":
        imputer = SimpleImputer(strategy="constant", fill_value="missing")
    else:
        raise ValueError(f"Неизвестный impute: {impute}")

    # Кодирование признаков
    if encoding == "onehot":
        encoder = OneHotEncoder(handle_unknown="ignore")
    elif encoding == "target":
        encoder = TargetEncoder(random_state=random_state)
    elif encoding == "ordinal":
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
    else:
        raise ValueError(f"Неизвестный encoding: {encoding}")

    return Pipeline(
        [
            ("impute", imputer),
            ("encode", encoder),
        ]
    )


def build_pipeline(cfg: DictConfig) -> Pipeline:
    """Собирает пайплайн из препроцессинга и модели."""
    preprocessor = ColumnTransformer(
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
    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", instantiate(cfg.model)),
        ]
    )


def fit_cv(
    cfg: DictConfig, X: pd.DataFrame, y: pd.Series
) -> tuple[list[Pipeline], np.ndarray, np.ndarray, str]:
    """Считает CV, возвращает модели фолдов, OOF-вероятности класса 1, скоры и метрику."""
    cv = instantiate(cfg.validation.cv)
    metric = cfg.validation.metric
    pipe = build_pipeline(cfg)
    res = cross_validate(pipe, X, y, cv=cv, scoring=metric, return_estimator=True)

    # out-of-fold (oof): сохраняем вероятность класса 1 для объектов, на которых каждая модель не училась. Берем вероятность класса 1
    oof = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    scores = res["test_score"]  # предсказания по фолдам
    pipes = res["estimator"]  # n_splits моделей
    return pipes, oof, scores, metric


def run(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: DictConfig,
) -> tuple[list[Pipeline], np.ndarray, float]:
    """Прогоняет CV, логирует скор, возвращает модели фолдов, OOF-вероятности и средний скор."""
    config = cast(  # cast показывает тип pyrefly
        "dict[str, object]", OmegaConf.to_container(cfg, resolve=True)
    )

    with wandb.init(**cfg.experiment.wandb, config=config) as wb_run:
        pipes, oof, scores, metric = fit_cv(cfg, X, y)
        wb_run.log({f"cv_{metric}": scores.mean(), f"cv_{metric}_std": scores.std()})

    return pipes, oof, scores.mean()


def predict(pipes: list[Pipeline], X: pd.DataFrame) -> np.ndarray:
    """Усредняет вероятности моделей фолдов и возвращает вероятность класса 1."""
    probs = [pipe.predict_proba(X)[:, 1] for pipe in pipes]
    return np.mean(probs, axis=0)
