import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Настройка страницы
st.set_page_config(page_title="Прогноз ACHP растений", layout="wide")
st.title("Прогноз ACHP растений")

# Создание боковой панели с настройками модели
st.sidebar.header("Настройки модели")
n_estimators = st.sidebar.slider("Количество деревьев", 50, 200, 100)
max_depth = st.sidebar.slider("Максимальная глубина дерева", 1, 20, 10)
min_samples_split = st.sidebar.slider("Минимальное кол-во для разделения", 2, 10, 2)

# Функция для загрузки данных и моделей
@st.cache_data
def load_data():
    data_path = os.path.join('Processed_Plant_Growth_Metrics.csv')
    df = pd.read_csv(data_path)
    # Проверка наличия столбцов с полностью пропущенными значениями
    null_columns = df.columns[df.isnull().all()].tolist()
    if null_columns:
        print(f"Столбцы с полностью пропущенными значениями: {null_columns}")
        df = df.drop(columns=null_columns)
    X = df.drop('remainder__ACHP', axis=1)
    y = df['remainder__ACHP']
    return df, X, y

# Загрузка данных и предварительно обученных моделей
df, X, y = load_data()
scaler = joblib.load('scaler.pkl')
imputer = joblib.load('imputer.pkl')
default_model = joblib.load('best_randomforest.pkl')

# Загрузка списка признаков
try:
    feature_names = joblib.load('feature_names.pkl')
    print("Загружен список признаков:", len(feature_names))
except:
    feature_names = X.columns.tolist()
    print("Использован список признаков из текущих данных:", len(feature_names))

# Функция для визуализации важности признаков
def plot_feature_importance(model, feature_names):
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    fig, ax = plt.figure(figsize=(10, 6)), plt.subplot(111)
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10), ax=ax)
    ax.set_title('Важность признаков (RandomForest)')
    return fig

# Отображение изображений сравнения моделей и важности признаков
st.sidebar.header("Визуализация моделей")
show_comparison = st.sidebar.checkbox("Показать сравнение моделей")
if show_comparison:
    st.sidebar.image('models_comparison.png', caption='Сравнение моделей')

# Функция для обучения новой модели с выбранными параметрами
def train_model(n_estimators, max_depth, min_samples_split):
    # Загрузка данных для обучения
    _, X_train, y_train = load_data()
    
    # Разделение на обучающую и тестовую выборки
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Обработка пропущенных значений
    X_train_imputed = imputer.transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Масштабирование данных
    X_train_scaled = scaler.transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Создание и обучение модели
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Оценка качества модели
    y_pred = model.predict(X_test_scaled)
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }
    
    return model, metrics

# Создание 2 колонок: настройки и результаты
col1, col2 = st.columns([1, 2])

# Колонка настроек
with col1:
    st.header("Настройки прогноза")
    # Создание формы для ввода данных
    input_data = {}
    feature_cols = [col for col in X.columns if col in feature_names]
    
    for i, col in enumerate(feature_cols):
        input_data[col] = st.number_input(
            f"{col}",
            min_value=float(df[col].min()),
            max_value=float(df[col].max()),
            value=float(df[col].mean())
        )
    
    # Выбор режима модели
    use_custom_model = st.checkbox("Использовать модель с настраиваемыми параметрами")
    
    if use_custom_model:
        if st.button("Обучить модель с новыми параметрами"):
            with st.spinner("Обучаем модель..."):
                custom_model, model_metrics = train_model(n_estimators, max_depth, min_samples_split)
                st.session_state.custom_model = custom_model
                st.session_state.model_metrics = model_metrics
                st.session_state.model_trained = True
                st.success("Модель обучена!")
    else:
        st.session_state.model_trained = False

# Колонка результатов
with col2:
    st.header("Результаты и анализ")
    
    # Отображение метрик модели если модель была обучена
    if use_custom_model and 'model_trained' in st.session_state and st.session_state.model_trained:
        st.subheader("Метрики модели с выбранными параметрами")
        metrics = st.session_state.model_metrics
        metrics_df = pd.DataFrame({
            'Метрика': list(metrics.keys()),
            'Значение': list(metrics.values())
        })
        st.table(metrics_df)
        
        # Вывод важности признаков для новой модели
        st.subheader("Важность признаков")
        importance_fig = plot_feature_importance(st.session_state.custom_model, feature_names)
        st.pyplot(importance_fig)
    else:
        # Отображение предварительно рассчитанной важности признаков
        st.subheader("Важность признаков (предобученная модель)")
        st.image('feature_importance.png')
    
    # Кнопка для получения прогноза
    st.header("Получить прогноз")
    if st.button("Сделать прогноз"):
        # Преобразование входных данных
        input_df = pd.DataFrame([input_data])
        # Убедимся, что порядок столбцов соответствует порядку при обучении
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0  # Заполняем отсутствующие столбцы нулями
        
        input_df = input_df[feature_names]  # Упорядочиваем столбцы
        input_imputed = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imputed)
        
        # Выбор модели для прогнозирования
        if use_custom_model and 'model_trained' in st.session_state and st.session_state.model_trained:
            prediction = st.session_state.custom_model.predict(input_scaled)
            model_type = "пользовательской моделью"
        else:
            prediction = default_model.predict(input_scaled)
            model_type = "предобученной моделью"
        
        # Вывод результата
        st.success(f"Прогнозируемый ACHP с {model_type}: {prediction[0]:.4f}")

# Добавление информации о модели и проекте
st.sidebar.header("О проекте")
st.sidebar.info("""
### Анализ роста растений

Этот проект использует машинное обучение для прогнозирования параметров роста растений в теплице.

**Использованные модели:**
- LinearRegression
- Ridge 
- Lasso
- RandomForest
- GradientBoosting

**Выводы:**
- Ансамблевые методы (RandomForest и GradientBoosting) показали наилучшие результаты
- R2 для RandomForest составил около 0.94, что указывает на высокую точность модели
- Наиболее важными признаками являются ...

**Инструкция:**
1. Выберите параметры модели в боковой панели
2. Настройте параметры растения в левой колонке
3. Обучите модель с новыми параметрами или используйте предобученную
4. Нажмите "Сделать прогноз" для получения результатов
""") 