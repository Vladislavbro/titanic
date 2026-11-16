from titanic.config import load_config
from titanic.data import prepare_data, save_submission
from titanic.train import run


def main() -> None:
    cfg = load_config()
    X_train, X_test, y_train, test_ids = prepare_data(cfg)
    pipe = run(X_train, y_train, cfg)
    save_submission(test_ids, pipe.predict(X_test), cfg)


if __name__ == "__main__":
    main()
