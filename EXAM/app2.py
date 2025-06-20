import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV

# Загрузка данных
data = pd.read_csv('dataset_01.csv', sep=';')
X = data[['x1', 'x2', 'x3']]
y = data['y']

# Создаем pipeline с масштабированием и SVR
model = make_pipeline(
    StandardScaler(),
    SVR(kernel='rbf')
)

# Параметры для оптимизации
param_grid = {
    'svr__C': [0.1, 1, 10, 100],
    'svr__gamma': [0.01, 0.1, 1, 10],
    'svr__epsilon': [0.01, 0.05, 0.1, 0.2]
}

# Поиск лучших параметров
grid = GridSearchCV(model, param_grid, cv=5, scoring='neg_max_error', n_jobs=-1)
grid.fit(X, y)

# Лучшая модель
best_model = grid.best_estimator_

# Проверка метрики для лучшей модели
scores = cross_val_score(best_model, X, y, cv=5, scoring='neg_max_error')
mean_max_error = np.mean(np.abs(scores))

print(f"Лучшие параметры: {grid.best_params_}")
print(f"Усредненный max_error: {mean_max_error:.6f}")

if mean_max_error <= 0.22:
    best_model.fit(X, y)
    print("Модель удовлетворяет условию!")
else:
    print("Дополнительный анализ данных необходим")