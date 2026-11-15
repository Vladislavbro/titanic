from pathlib import Path

from omegaconf import DictConfig, OmegaConf

CONFIG_PATH = Path(__file__).parents[1] / "config.yaml"


def load_config() -> DictConfig:
    cfg = OmegaConf.load(CONFIG_PATH)
    OmegaConf.set_struct(cfg, True)  # запрет новых ключей
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    OmegaConf.set_readonly(cfg, True)  # запрет изменений
    return cfg
