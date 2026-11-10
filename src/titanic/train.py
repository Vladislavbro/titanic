from omegaconf import OmegaConf
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier

cfg = OmegaConf.merge(
    OmegaConf.load("config.yaml"),
    OmegaConf.from_cli(),        # переопределения из командной строки
)

if cfg.model.name == "rf":
    rf_params = cfg.model.rf
    model = RandomForestClassifier(n_estimators=rf_params.n_estimators,
                                   max_depth=rf_params.max_depth,
                                   random_state=cfg.seed)