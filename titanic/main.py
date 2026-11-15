from titanic.config import load_config
from titanic.data import load_data
from titanic.features import make_features


def main():
    cfg = load_config()
    X_train, X_test, y_train = load_data(cfg)
    test_ids = X_test[cfg.data.id_col]
    X_train = make_features(X_train, cfg.data)
    X_test = make_features(X_test, cfg.data)

    


if __name__ == "__main__":
    main()
