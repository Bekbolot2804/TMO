import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Загрузка данных
data_path = os.path.join('Processed_Plant_Growth_Metrics.csv')
df = pd.read_csv(data_path)

# Проверка наличия столбцов с полностью пропущенными значениями
null_columns = df.columns[df.isnull().all()].tolist()
if null_columns:
    print(f"Столбцы с полностью пропущенными значениями: {null_columns}")
    df = df.drop(columns=null_columns)  # Удаляем столбцы с полностью пропущенными значениями

# Предобработка данных
X = df.drop('remainder__ACHP', axis=1)  # Удаляем целевую переменную
y = df['remainder__ACHP']  # Целевая переменная

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обработка пропущенных значений
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Масштабирование данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Сохранение scaler и imputer
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(imputer, 'imputer.pkl')

# Определение моделей
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'RandomForest': RandomForestRegressor(random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42)
}

# Обучение базовых моделей
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    results[name] = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }

# Вывод результатов базовых моделей
print("Результаты базовых моделей:")
print(pd.DataFrame(results).T)

# Подбор гиперпараметров для лучших моделей
param_grid = {
    'RandomForest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    },
    'GradientBoosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5, 7]
    }
}

best_models = {}
for name in ['RandomForest', 'GradientBoosting']:
    grid = GridSearchCV(
        models[name],
        param_grid[name],
        cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    grid.fit(X_train_scaled, y_train)
    best_models[name] = grid.best_estimator_
    print(f"\nЛучшие параметры для {name}:")
    print(grid.best_params_)

# Сохранение лучших моделей
for name, model in best_models.items():
    joblib.dump(model, f'best_{name.lower()}.pkl')

# Визуализация результатов
plt.figure(figsize=(12, 6))
df_results = pd.DataFrame(results).T
df_results[['MAE', 'MSE', 'R2']].plot(kind='bar')
plt.title('Сравнение моделей по метрикам')
plt.tight_layout()
plt.savefig('models_comparison.png')
plt.close()

# Сохраняем список признаков, которые использовались для обучения
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'feature_names.pkl')

# Визуализация важности признаков для RandomForest
try:
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': best_models['RandomForest'].feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(15))  # Показываем только 15 наиболее важных признаков
    plt.title('Важность признаков (RandomForest)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    print("Визуализация важности признаков создана успешно")
except Exception as e:
    print(f"Ошибка при создании визуализации важности признаков: {e}")
    print("Длина feature_names:", len(feature_names))
    print("Длина feature_importances_:", len(best_models['RandomForest'].feature_importances_)) 