import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV

# Загрузка данных
data = pd.read_csv('dataset_01.csv', sep=';')
X = data[['x1', 'x2', 'x3']]
y = data['y']

# Попробуем простую линейную регрессию
model = make_pipeline(
    StandardScaler(),
    LinearRegression()
)

# Проверка метрики
scores = cross_val_score(model, X, y, cv=5, scoring='neg_max_error')
mean_max_error = np.mean(np.abs(scores))
print(f"Простая линейная регрессия - Усредненный max_error: {mean_max_error:.6f}")

if mean_max_error <= 0.22:
    print("Простая линейная регрессия удовлетворяет условию!")
    model.fit(X, y)
else:
    # Если не удовлетворяет - используем Ridge с подбором параметров
    print("Пробуем Ridge-регрессию с подбором параметров...")
    
    # Создаем pipeline для Ridge
    ridge_pipe = make_pipeline(
        StandardScaler(),
        Ridge()
    )
    
    # Параметры для GridSearchCV
    param_grid = {'ridge__alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    
    # Поиск лучших параметров
    grid = GridSearchCV(ridge_pipe, param_grid, cv=5, scoring='neg_max_error')
    grid.fit(X, y)
    
    # Лучшая модель
    best_model = grid.best_estimator_
    
    # Проверка метрики для лучшей модели
    scores = cross_val_score(best_model, X, y, cv=5, scoring='neg_max_error')
    mean_max_error = np.mean(np.abs(scores))
    
    print(f"Лучший alpha: {grid.best_params_['ridge__alpha']}")
    print(f"Ridge регрессия - Усредненный max_error: {mean_max_error:.6f}")
    
    if mean_max_error <= 0.22:
        best_model.fit(X, y)
        print("Ridge-регрессия удовлетворяет условию!")
    else:
        print("Требуется более сложный подход")