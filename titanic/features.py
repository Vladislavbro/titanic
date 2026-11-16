import numpy as np
import pandas as pd
from omegaconf import DictConfig


def make_features(df: pd.DataFrame, cfg_data: DictConfig) -> pd.DataFrame:
    """Создание и отбор фичей."""
    df = df.copy()
    columns = list(cfg_data.num_features) + list(cfg_data.cat_features)

    if "LogFare" in columns:
        df["LogFare"] = np.log1p(df["Fare"])

    if "Title" in columns:
        df["Title"] = df["Name"].str.extract(r",\s*([^.]+)\.")[0]

    if "Deck" in columns:
        df["Deck"] = df["Cabin"].str[0]

    if "FamilySize" in columns:
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["FamilySize"] = pd.cut(
            df["FamilySize"], bins=[0, 1, 4, 100], labels=["alone", "small", "large"]
        )

    return df[columns]
