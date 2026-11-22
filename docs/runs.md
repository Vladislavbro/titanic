# Запуски

Все команды выполняются из корня репозитория после `uv sync`.

## Линейные модели

```bash
uv run main.py -m model=logreg model.C=0.01,0.1,1,10
uv run main.py -m model=logreg_l1 model.C=0.01,0.1,1
uv run main.py -m model=logreg_elasticnet model.l1_ratio=0.2,0.5,0.8
```

## KNN

```bash
uv run main.py -m model=knn model.n_neighbors=3,5,11,21 model.weights=uniform,distance
```

## Деревья

```bash
uv run main.py -m model=tree model.max_depth=3,5,7
uv run main.py -m model=rf model.max_depth=5,7
```

## Бустинги

```bash
uv run main.py -m model=catboost model.depth=2,3,4,6
uv run main.py -m model=catboost model.learning_rate=0.03,0.05,0.1 model.iterations=300,1000

uv run main.py -m model=lightgbm model.num_leaves=15,31,63
uv run main.py -m model=lightgbm model.learning_rate=0.03,0.05,0.1

uv run main.py -m model=xgboost model.max_depth=3,4,6
uv run main.py -m model=xgboost model.learning_rate=0.03,0.05,0.1
```

## Нейросеть

```bash
uv run main.py model=mlp
uv run main.py -m model=mlp model.hidden_sizes=[64],[64,32],[64,32,16]
uv run main.py -m model=mlp model.activation=relu,gelu,tanh model.dropout=0,0.3,0.5
uv run main.py -m model=mlp model.batchnorm=true,false model.optimizer=adamw,adam,sgd
uv run main.py -m model=mlp model.scheduler=none,cosine model.lr=1e-2,1e-3
```

Руками, без CV и wandb, на разбиении 80/20 — `scripts/sandbox_mlp.py`:

```bash
uv run python -m scripts.sandbox_mlp
```

Печатает лосс, auc и accuracy по эпохам, пишет `submissions/sandbox_mlp.csv`.

## Optuna

```bash
uv run main.py -m model=catboost hydra/sweeper=optuna_catboost
uv run main.py -m model=lightgbm hydra/sweeper=optuna_lightgbm
uv run main.py -m model=xgboost  hydra/sweeper=optuna_xgboost
```

Триалы копятся в `optuna.db`, прерванный перебор продолжается с того же места.

## Ансамбли

`blend.py` читает `submissions/*__oof.csv` и `*__test_proba.csv`, поэтому сначала прогоняются базовые модели. Список моделей, порог, варианты смешивания и мета-модели — `configs/ensemble.yaml`.

```bash
uv run blend.py
uv run blend.py ensemble.models=[catboost,xgboost,mlp] ensemble.variants=[mean,stack_logreg]
```

Варианты: `mean` — среднее вероятностей, `vote` — большинство голосов, `stack_logreg` и `stack_ridge` — мета-модель поверх OOF. Скор считается по OOF, сабмиты — `submissions/ensemble_<вариант>.csv`.

## CatBoost с early stopping

Число деревьев подбирается early stopping по фолдам, затем модель обучается на всём трейне и предсказывает тест один раз.

```bash
uv run python -m scripts.catboost_earlystop
```

Сабмит — `submissions/catboost_earlystop.csv`.
