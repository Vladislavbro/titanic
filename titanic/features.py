from typing import Self, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

COMMON_TITLES = {"Mr", "Mrs", "Miss", "Master"}
TITLE_ALIASES = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}


def make_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Добавляет статичные фичи в train и test."""
    df = pd.concat([X_train, X_test], ignore_index=True)

    df["LogFare"] = np.log1p(df["Fare"])

    title = (
        df["Name"].str.extract(r",\s*([^.]+)\.")[0].replace(TITLE_ALIASES)
    )  # replace (а не map), чтобы остальные титулы не стали NaN
    df["Title"] = title.where(title.isin(COMMON_TITLES), "Rare")

    df["Deck"] = df["Cabin"].str[0]
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    df["TicketGroupSize"] = df.groupby("Ticket")["Ticket"].transform("count")
    df["FarePerPerson"] = df["Fare"] / df["TicketGroupSize"]

    n_train = len(X_train)
    train_out = df.iloc[:n_train].copy().set_axis(X_train.index)
    test_out = df.iloc[n_train:].copy().set_axis(X_test.index)
    return train_out, test_out


# 1. BaseEstimator даёт методы get_params / set_params,
#    чтобы cross_validate мог пересоздавать объекты класса для каждого фолда.
# 2. TransformerMixin добавляет метод fit_transform(X, y), который вызывает fit, потом transform.
#    Нужен Pipeline.
class FoldFeatures(BaseEstimator, TransformerMixin):
    """Заполняет пропуски возраста внутри фолдов."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        self.age_fallback_: float = X["Age"].median()
        self.age_medians_ = cast(  # cast подсказывает формат тайпчекеру (и читателю)
            dict[tuple[str, int], float],
            X.groupby(["Title", "Pclass"])["Age"].median().to_dict(),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for (title, pclass), median in self.age_medians_.items():
            mask = (
                (X["Age"].isna()) & (X["Title"] == title) & (X["Pclass"] == pclass)
            )  # маска по 3 условиям
            X.loc[mask, "Age"] = median  # все строки одна колонка
        X["Age"] = X["Age"].fillna(self.age_fallback_)
        return X
