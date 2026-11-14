from omegaconf import DictConfig
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

MODELS = {
    "logreg": LogisticRegression,
    "rf": RandomForestClassifier,
}


def build_model(cfg: DictConfig) -> LogisticRegression | RandomForestClassifier:
    return MODELS[cfg.model.name](
        **cfg.model.params, random_state=cfg.seed
    )  # TODO: поправить кривизну
