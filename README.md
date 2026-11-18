# Titanic

## Установка

```bash
uv sync
```

## Запуск

Одна модель:

```bash
uv run main.py model=knn
```

Все модели подряд:

```bash
uv run main.py -m model=logreg,logreg_l1,logreg_elasticnet,knn,tree,rf
```

Перебор гиперпараметров:

```bash
uv run main.py -m model=knn model.n_neighbors=3,5,11,21 model.weights=uniform,distance
uv run main.py -m model=logreg model.C=0.01,0.1,1,10

uv run main.py -m model=tree model.max_depth=3,5,7
uv run main.py -m model=rf model.max_depth=5,7

uv run main.py -m model=catboost model.depth=2,3,4,6
uv run main.py -m model=catboost model.learning_rate=0.03,0.05,0.1 model.iterations=300,1000
uv run main.py -m model=catboost model.l2_leaf_reg=1,3,10
uv run main.py -m model=catboost model.random_strength=1,5 model.bagging_temperature=0,1


```

`-m` прогоняет все комбинации по очереди. Конфиги моделей лежат в `configs/model/`, скоры логируются в wandb.

Перебор через Optuna:

```bash
uv run main.py -m model=catboost hydra/sweeper=optuna_catboost
uv run main.py -m model=lightgbm hydra/sweeper=optuna_lightgbm
uv run main.py -m model=xgboost  hydra/sweeper=optuna_xgboost
```

Триалы копятся в `optuna.db`, прерванный перебор продолжается с того же места.
