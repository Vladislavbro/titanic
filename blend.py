import hydra
from omegaconf import DictConfig

from titanic.ansembles import run


@hydra.main(config_path="configs", config_name="ensemble", version_base=None)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
