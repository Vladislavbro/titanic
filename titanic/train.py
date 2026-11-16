import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    TargetEncoder,
)

import wandb


def make_num_pipeline(impute: str, add_flag: bool, scale: bool) -> Pipeline:
    """Числовые признаки: заполнение пропусков и (опционально) скейлинг."""
    # Заполнение пропусков
    imputer = SimpleImputer(
        strategy=impute,
        add_indicator=add_flag,
    )
    steps = [("impute", imputer)]

    # Масштабирование
    if scale:
        steps.append(("scale", StandardScaler()))

    return Pipeline(steps)


def make_cat_pipeline(impute: str, encoding: str, random_state: int) -> Pipeline:
    """Категориальные признаки: заполнение пропусков и кодирование."""
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
    """Препроцессинг + модель."""
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


def run(X: pd.DataFrame, y: pd.Series, cfg: DictConfig) -> Pipeline:
    """CV-оценка с логированием в wandb и обучение на всём train."""
    pipe = build_pipeline(cfg)
    cv = instantiate(cfg.validation.cv)
    metric = cfg.validation.metric
    config = OmegaConf.to_container(cfg, resolve=True)

    with wandb.init(**cfg.experiment.wandb, config=config) as wb_run:
        scores = cross_val_score(pipe, X, y, cv=cv, scoring=metric)
        wb_run.log({f"cv_{metric}": scores.mean(), f"cv_{metric}_std": scores.std()})

    return pipe.fit(X, y)
