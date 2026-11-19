from torch._tensor import Tensor

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from titanic.data import prepare_data

device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)


# Структура:
#   - На вход модель получает фолды из cv: 80/20 трейн и тест после препроцессинга
#   - Надо разбить данные на train val
#   - Обучить модель

train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=64)
val_loader = DataLoader(val_ds, batch_size=64)


# ===== Модель: выдаёт ЛОГИТЫ, без Sigmoid на конце =====
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 128),
    nn.ReLU(),
    nn.Linear(128, 1),
).to(device)

criterion = nn.BCEWithLogitsLoss()  # sigmoid + BCE, то есть logloss
optimizer = optim.AdamW(
    model.parameters(), lr=1e-3, weight_decay=1e-2
)  # model.parameters() - это w и b слоев


# ===== Обучение =====
for epoch in range(10):
    model.train()
    for xb, yb in train_loader:  # возращает X и y батчами
        xb, yb = xb.to(device), yb.to(device)
        loss = criterion(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# ===== Оценка =====
model.eval()  # dropout/batchnorm в режиме инференса
batches = []  # сюда кладём выход модели по одному батчу

with torch.no_grad():  # граф не строим, памяти не тратим
    for xb, yb in val_loader:  # xb: (batch, n_features), yb: (batch, 1)
        out = model(xb.to(device))  # out: (batch, 1) — логиты
        batches.append(out)

logits = torch.cat(batches)  # (N, 1) — все батчи в один тензор
proba = torch.sigmoid(logits)
proba = proba.cpu().numpy()
proba = proba.ravel()  # (N,) — плоский, как ждёт sklearn
