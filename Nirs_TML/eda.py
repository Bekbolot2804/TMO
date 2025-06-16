import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Загрузка данных с абсолютным путем
data_path = os.path.join('G:', 'ОАД', 'НИРС', 'Processed_Plant_Growth_Metrics.csv')
df = pd.read_csv(data_path)

# Удаляем пустой столбец
df = df.drop('remainder__AWR', axis=1)

# Вывод названий столбцов
print("\nНазвания столбцов в датасете:")
print(df.columns.tolist())

# Базовая информация о данных
print("\nИнформация о данных:")
print(df.info())
print("\nСтатистическое описание:")
print(df.describe())
print("\nПропущенные значения:")
print(df.isnull().sum())

# Визуализация пропущенных значений
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False)
plt.title('Визуализация пропущенных значений')
plt.tight_layout()
plt.savefig('missing_values.png')
plt.close()

# Гистограммы числовых признаков
df.hist(bins=30, figsize=(15, 10))
plt.tight_layout()
plt.savefig('histograms.png')
plt.close()

# Корреляционная матрица
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Корреляционная матрица')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Анализ корреляций для каждого признака
print("\nКорреляции между признаками:")
for column in df.columns:
    correlations = correlation_matrix[column].sort_values(ascending=False)
    print(f"\nКорреляции для {column}:")
    print(correlations[correlations != 1.0].head())

# Сохранение обработанных данных
df.to_csv('processed_data.csv', index=False) 