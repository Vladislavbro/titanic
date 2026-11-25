from pathlib import Path

import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from titanic.data import ROOT


def resolve_run(cfg: DictConfig, model: str) -> Path:
    """Находит папку прогона модели в submissions."""
    base = ROOT / cfg.data.submission_path
    runs = cfg.ensemble.get("runs")
    if runs is not None and model in runs:
        return base / runs[model]
    if (base / model).is_dir():
        return base / model
    candidates = sorted(path for path in base.glob(f"{model}_*") if path.is_dir())
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"нет прогонов модели {model} в {base}")
    names = ", ".join(path.name for path in candidates)
    raise ValueError(
        f"несколько прогонов модели {model}: {names};"
        f" выбери папку в ensemble.runs.{model}"
    )


def run_path(cfg: DictConfig, model: str, kind: str) -> Path:
    """Возвращает путь к файлу прогона модели."""
    return resolve_run(cfg, model) / f"{kind}.csv"


def load_oof(cfg: DictConfig) -> tuple[np.ndarray, np.ndarray]:
    """Собирает OOF-вероятности моделей и таргет, выравнивая по id."""
    frames = {
        model: pd.read_csv(run_path(cfg, model, "oof"), index_col=cfg.data.id_col)
        for model in cfg.ensemble.models
    }
    oof = pd.DataFrame({model: frame["proba"] for model, frame in frames.items()})
    y = next(iter(frames.values()))["y"].to_numpy()
    return oof.to_numpy(), y


def load_test(cfg: DictConfig) -> tuple[pd.Index, np.ndarray]:
    """Собирает id и вероятности моделей на тесте, выравнивая по id."""
    columns = {
        model: pd.read_csv(
            run_path(cfg, model, "test_proba"), index_col=cfg.data.id_col
        )["proba"]
        for model in cfg.ensemble.models
    }
    probs = pd.DataFrame(columns)
    return probs.index, probs.to_numpy()


def mean_variant(
    cfg: DictConfig, oof: np.ndarray, y: np.ndarray, test: np.ndarray
) -> tuple[float, np.ndarray]:
    """Усредняет вероятности моделей и возвращает OOF-точность и тестовые классы."""
    threshold = cfg.ensemble.threshold
    oof_preds = (oof.mean(axis=1) >= threshold).astype(int)
    return (
        float(accuracy_score(y, oof_preds)),
        (test.mean(axis=1) >= threshold).astype(int),
    )


def vote_variant(
    cfg: DictConfig, oof: np.ndarray, y: np.ndarray, test: np.ndarray
) -> tuple[float, np.ndarray]:
    """Выбирает класс большинством голосов и возвращает OOF-точность и тестовые классы."""
    threshold = cfg.ensemble.threshold
    half = 0.5
    oof_preds = ((oof >= threshold).mean(axis=1) > half).astype(int)
    return (
        float(accuracy_score(y, oof_preds)),
        ((test >= threshold).mean(axis=1) > half).astype(int),
    )


def stack_variant(
    cfg: DictConfig, name: str, oof: np.ndarray, y: np.ndarray, test: np.ndarray
) -> tuple[float, np.ndarray]:
    """Обучает мета-модель на OOF-вероятностях и возвращает CV-точность и тестовые классы."""
    meta = instantiate(cfg.ensemble.metas[name])
    cv = instantiate(cfg.validation.cv)
    score = float(
        cross_val_score(meta, oof, y, cv=cv, scoring=cfg.validation.metric).mean()
    )
    meta.fit(oof, y)
    return score, meta.predict(test)


def build_variant(
    cfg: DictConfig, name: str, oof: np.ndarray, y: np.ndarray, test: np.ndarray
) -> tuple[float, np.ndarray]:
    """Считает OOF-точность и тестовые предсказания для варианта ансамбля."""
    if name == "mean":
        return mean_variant(cfg, oof, y, test)
    if name == "vote":
        return vote_variant(cfg, oof, y, test)
    return stack_variant(cfg, name, oof, y, test)


def save_submission(
    cfg: DictConfig, name: str, ids: pd.Index, preds: np.ndarray
) -> None:
    """Сохраняет сабмит варианта ансамбля."""
    path = ROOT / cfg.data.submission_path / f"ensemble_{name}"
    path.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({cfg.data.id_col: ids, cfg.data.target_col: preds})
    out.to_csv(path / f"{path.name}.csv", index=False)


def run(cfg: DictConfig) -> pd.DataFrame:
    """Прогоняет варианты из конфига и сохраняет сабмиты с таблицей скоринга."""
    oof, y = load_oof(cfg)
    ids, test = load_test(cfg)

    rows = []
    for name in cfg.ensemble.variants:
        score, test_preds = build_variant(cfg, name, oof, y, test)
        save_submission(cfg, name, ids, test_preds)
        rows.append({"variant": name, "oof_accuracy": score})

    table = pd.DataFrame(rows).sort_values("oof_accuracy", ascending=False)
    table.to_csv(ROOT / cfg.data.submission_path / "ensemble_results.csv", index=False)
    print(table.to_string(index=False))
    return table
