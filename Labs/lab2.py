import pandas as pd
from sklearn.preprocessing import StandardScaler

# Загрузка данных
df = pd.read_csv('Greenhouse Plant Growth Metrics.csv')

# Обработка пропусков
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

if df['Class'].isnull().any():
    df['Class'] = df['Class'].fillna(df['Class'].mode()[0])

# Кодирование категориальных признаков
df = df.drop('Random', axis=1)
df = pd.get_dummies(df, columns=['Class'], prefix='Class')

# Масштабирование данных
scaler = StandardScaler()
numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Результат
print(df.head())