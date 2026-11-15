from pathlib import Path

from omegaconf import DictConfig, OmegaConf

CONFIGS = Path(__file__).parents[1] / "configs"


def load_config() -> DictConfig:
    configs = [OmegaConf.load(path) for path in sorted(CONFIGS.glob("*.yaml"))]
    cfg = OmegaConf.merge(*configs)

    OmegaConf.set_struct(cfg, True)  # запрет новых ключей
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    OmegaConf.set_readonly(cfg, True)  # запрет изменений
    return cfg
