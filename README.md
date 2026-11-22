# Titanic

Классификация выживших пассажиров: препроцессинг в sklearn-пайплайне, 5-fold CV, сравнение моделей и ансамбли.

## Стек

- Python 3.13, `uv`, hydra + omegaconf для конфигов, wandb для логов
- sklearn: препроцессинг, логистическая регрессия, KNN, деревья, Random Forest, стекинг
- CatBoost, LightGBM, XGBoost
- PyTorch: собственный `MLPClassifier` в стиле sklearn
- Метрика — accuracy, валидация — `StratifiedKFold(5)`

## Запуск

```bash
uv sync                      # окружение
uv run main.py               # один прогон, по умолчанию logreg
uv run main.py model=catboost
uv run main.py -m model=logreg,knn,tree,rf,catboost,lightgbm,xgboost,mlp   # все модели
uv run blend.py              # ансамбли поверх сохранённых OOF-вероятностей
```

Модель выбирается конфигом в `configs/model/`, гиперпараметры — полями этого конфига или оверрайдами (`uv run main.py model=rf model.max_depth=5`).

Что появляется в `submissions/`:

- `<модель>.csv` — сабмит для Kaggle
- `<модель>__oof.csv` — out-of-fold вероятности (используются блендером)
- `<модель>__test_proba.csv` — вероятности на тесте
- `ensemble_<вариант>.csv` и `ensemble_results.csv` — ансамбли и их OOF-скор

wandb по умолчанию работает offline; для синхронизации — `WANDB_MODE=online uv run main.py`.

Остальные запуски (перебор гиперпараметров, Optuna, MLP, ансамбли, CatBoost с early stopping) — в [docs/runs.md](docs/runs.md).

## Результаты

| модель | CV accuracy | LB |
| --- | --- | --- |
| catboost | 0.8429 | — |
| xgboost | 0.8406 | — |
| lightgbm | 0.8350 | — |
| rf | 0.8294 | — |
| logreg | 0.8272 | — |
| mlp | 0.8215 | — |
| knn | 0.8125 | — |
| ensemble stack_logreg | 0.8507 | 0.77511 |
| ensemble mean | 0.8496 | 0.77033 |
| ensemble vote | 0.8384 | 0.77033 |
| catboost_earlystop | 0.8361 | 0.77272 |
| baseline «женщины выживают» | 0.7868 | 0.76555 |
