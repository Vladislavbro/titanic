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
```

`-m` прогоняет все комбинации по очереди. Конфиги моделей лежат в `configs/model/`, скоры логируются в wandb.
