import numpy as np
import pandas as pd

COMMON_TITLES = {"Mr", "Mrs", "Miss", "Master"}
TITLE_ALIASES = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}


def make_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Добавляет статичные фичи в train и test."""
    df = pd.concat([X_train, X_test], ignore_index=True)

    df["LogFare"] = np.log1p(df["Fare"])

    title = df["Name"].str.extract(r",\s*([^.]+)\.")[0].replace(TITLE_ALIASES)
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
