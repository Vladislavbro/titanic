from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from omegaconf import DictConfig
from sklearn.pipeline import Pipeline

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
    X_train, X_test = make_features(X_train, X_test)
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


def _extract_importance(model: object) -> np.ndarray | None:
    """Извлекает важность признаков из обученной модели."""
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)
    if hasattr(model, "coef_"):
        return np.abs(np.asarray(model.coef_, dtype=float)).ravel()
    return None


def save_feature_importance(
    pipes: list[Pipeline],
    cfg: DictConfig,
    name: str | None = None,
) -> None:
    """Сохраняет таблицу и график важности признаков по фолдам."""
    folds: dict[str, pd.Series] = {}
    for idx, pipe in enumerate(pipes):
        imp = _extract_importance(pipe.named_steps["model"])
        if imp is None:
            return
        names = pipe.named_steps["preprocess"].get_feature_names_out()
        folds[f"fold_{idx}"] = pd.Series(imp, index=names)

    fold_df = pd.DataFrame(folds).fillna(0.0)
    table = pd.DataFrame(
        {
            "feature": fold_df.index,
            "mean": fold_df.mean(axis=1),
            "std": fold_df.std(axis=1, ddof=0),
        }
    )
    for col in fold_df.columns:
        table[col] = fold_df[col]
    table = table.sort_values("mean", ascending=False).reset_index(drop=True)

    path = run_dir(cfg, name)
    table.to_csv(path / "feature_importance.csv", index=False)

    fig = Figure(figsize=(8, max(4, len(table) * 0.35)))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    ax.barh(table["feature"], table["mean"], xerr=table["std"])
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(path.name)
    fig.tight_layout()
    fig.savefig(path / "feature_importance.png", dpi=150)
