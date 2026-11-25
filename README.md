# Titanic

Классификация выживших пассажиров: препроцессинг и фичи в sklearn-пайплайне, валидация `RepeatedStratifiedKFold(5×10)` по accuracy, сравнение классических моделей, бустингов, PyTorch MLP, TabPFN и ансамблей.

Разведочный анализ и выводы:
1. [notebooks/sandbox.ipynb](notebooks/sandbox.ipynb).
2. [docs/notes.md](docs/notes.md)

## Структура

- `main.py` — запуск обучения, CV и генерации сабмита
- `blend.py` — ансамбли (mean, voting, stacking) поверх сохранённых OOF-вероятностей
- `titanic/` — загрузка данных, генерация фич, пайплайн обучения, PyTorch `MLPClassifier` и ансамбли
- `configs/` — Hydra-конфиги: общие настройки, модели (`configs/model/`), Optuna и ансамбли
- `docs/runs.md` — подробные команды перебора гиперпараметров, Optuna и блендинга

## Запуск

```bash
uv sync                          # установить окружение (Python 3.13)
uv run main.py                   # базовый прогон (logreg)
uv run main.py model=catboost    # любая модель из configs/model/
uv run blend.py                  # ансамбли поверх OOF-предсказаний
```

Параметры моделей и фичей переопределяются через Hydra (`uv run main.py model=rf model.max_depth=5`), результаты и сабмиты сохраняются в `submissions/`. Логирование в wandb по умолчанию offline (`WANDB_MODE=online` для синхронизации).

## Результаты

| Модель / подход | CV accuracy | Kaggle LB | Верно из 418 |
| --- | --- | --- | --- |
| TabPFN | — | **0.79665** | 333 |
| CatBoost (`depth=3, l2_leaf_reg=10`) | 0.8260 | 0.77990 | 326 |
| Ансамбль (`stack_logreg`) | 0.8507 | 0.77511 | 324 |
| XGBoost | 0.8473 | 0.77511 | 324 |
| Baseline «женщины выживают» | 0.7868 | 0.76555 | 320 |
| LightGBM | 0.8530 | 0.75358 | 315 |

На 891 строке высокий CV плохо переносится на LB: более сложные деревья и бустинги с CV ~0.85 переобучаются под трейн и проигрывают на тесте простому бейзлайну. Лучше всего сработали TabPFN и сильно регуляризованный неглубокий CatBoost.
