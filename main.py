import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from titanic.data import prepare_data, save_oof, save_submission
from titanic.train import predict, run

load_dotenv()


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> float:
    X_train, X_test, y_train, train_ids, test_ids = prepare_data(cfg)
    pipes, oof, score = run(X_train, y_train, cfg)
    preds = predict(pipes, X_test)
    save_oof(cfg, train_ids, y_train, oof)
    save_submission(test_ids, preds, cfg)

    return score


if __name__ == "__main__":
    main()
