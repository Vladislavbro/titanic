# Titanic

Классификация выживших пассажиров: препроцессинг в sklearn-пайплайне, 5-fold CV, сравнение моделей и ансамбли.

Разведочный анализ (распределения, пропуски, выживаемость по сегментам) и выводы — в [notebooks/sandbox.ipynb](notebooks/sandbox.ipynb).

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

| сабмит | CV accuracy | LB | человек из 418 |
| --- | --- | --- | --- |
| tabpfn_playground | — | **0.79665** | 333 |
| holdout 80/20 | — | 0.77751 | 325 |
| catboost_pclass | — | 0.77751 | 325 |
| ensemble stack_logreg | 0.8507 | 0.77511 | 324 |
| xgboost (5 фолдов) | 0.8473 | 0.77511 | 324 |
| catboost_earlystop | 0.8361 | 0.77272 | 323 |
| ensemble mean | 0.8496 | 0.77033 | 322 |
| ensemble vote | 0.8384 | 0.77033 | 322 |
| catboost depth=3 | 0.8507 | 0.76794 | 321 |
| catboost (1) | 0.8563 | 0.76555 | 320 |
| baseline «женщины выживают» | 0.7868 | 0.76555 | 320 |
| catboost depth=4 | 0.8530 | 0.75837 | 317 |
| ensemble stack_ridge | 0.8440 | 0.75837 | 317 |
| lightgbm | 0.8530 | 0.75358 | 315 |

CV не ранжирует LB: модель с CV .8563 дала столько же, сколько правило «женщины выживают», а лучший LB — у TabPFN.
