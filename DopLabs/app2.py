# fraud_detection_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

# Улучшенная загрузка данных с обработкой ошибок
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("upi_transactions_2024.csv")
        st.success("Данные успешно загружены!")
        return df
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {e}")
        st.stop()

# Улучшенная предобработка данных
def preprocess_data(df):
    # Проверка наличия необходимых колонок
    required_columns = ['transaction id', 'timestamp', 'fraud_flag']
    for col in required_columns:
        if col not in df.columns:
            st.error(f"Отсутствует обязательная колонка: {col}")
            st.stop()
    
    df = df.drop(['transaction id', 'timestamp'], axis=1, errors='ignore')
    
    # Преобразование категориальных переменных
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if col != 'fraud_flag':  # Целевая переменная не преобразовывается
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    return df

# Основной интерфейс приложения
def main():
    st.set_page_config(page_title="Fraud Detection System", layout="wide")
    st.title("🔍 Система обнаружения мошеннических транзакций UPI")
    st.markdown("Сравнение производительности ML моделей для выявления мошенничества")
    
    # Загрузка данных
    df = load_data()
    
    # Показать сырые данные
    with st.expander("Просмотр сырых данных"):
        st.dataframe(df.head())
        st.write(f"Размер данных: {df.shape[0]} строк, {df.shape[1]} колонок")
    
    # Предобработка
    df_processed = preprocess_data(df)
    
    # Показать обработанные данные
    with st.expander("Просмотр обработанных данных"):
        st.dataframe(df_processed.head())
        st.write("Типы данных после обработки:")
        st.write(df_processed.dtypes)
    
    # Разделение данных
    X = df_processed.drop('fraud_flag', axis=1)
    y = df_processed['fraud_flag']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Боковая панель для настроек
    st.sidebar.header("⚙️ Настройки моделей")
    
    # Выбор модели
    model_name = st.sidebar.selectbox(
        "Выберите модель",
        ["Logistic Regression", "Random Forest", "Gradient Boosting"]
    )
    
    # Гиперпараметры
    params = {}
    if model_name == "Random Forest":
        params['n_estimators'] = st.sidebar.slider("Число деревьев", 10, 200, 100)
        params['max_depth'] = st.sidebar.slider("Макс. глубина", 2, 50, 10)
        params['class_weight'] = 'balanced' if st.sidebar.checkbox("Балансировка классов", True) else None
    elif model_name == "Gradient Boosting":
        params['n_estimators'] = st.sidebar.slider("Число деревьев", 10, 200, 100)
        params['learning_rate'] = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1)
        params['subsample'] = st.sidebar.slider("Subsample", 0.1, 1.0, 0.8)
    elif model_name == "Logistic Regression":
        params['max_iter'] = st.sidebar.slider("Макс. итераций", 100, 2000, 1000)
        params['class_weight'] = 'balanced' if st.sidebar.checkbox("Балансировка классов", True) else None
    
    # Обучение модели
    if st.sidebar.button("🚀 Обучить модель", type="primary"):
        with st.spinner('Обучение модели...'):
            try:
                # Создание модели
                if model_name == "Logistic Regression":
                    model = LogisticRegression(**params)
                elif model_name == "Random Forest":
                    model = RandomForestClassifier(**params)
                elif model_name == "Gradient Boosting":
                    model = GradientBoostingClassifier(**params)
                
                # Обучение
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                
                # Отображение результатов
                st.success("Модель успешно обучена!")
                
                # Отчет классификации
                st.subheader(f"📊 Результаты модели: {model_name}")
                st.code(classification_report(y_test, y_pred))
                
                # Матрица ошибок
                st.subheader("📉 Матрица ошибок")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                            annot_kws={"size": 16}, 
                            xticklabels=['Законные', 'Мошенничество'],
                            yticklabels=['Законные', 'Мошенничество'])
                ax.set_xlabel('Предсказанные')
                ax.set_ylabel('Фактические')
                ax.set_title('Матрица ошибок')
                st.pyplot(fig)
                plt.close(fig)
                
                # ROC-кривая
                st.subheader("📈 ROC-кривая")
                fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(fpr, tpr, color='darkorange', lw=2, 
                         label=f'ROC кривая (AUC = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('ROC-кривая')
                ax.legend(loc="lower right")
                st.pyplot(fig)
                plt.close(fig)
                
                # Важность признаков
                if hasattr(model, 'feature_importances_'):
                    st.subheader("🔍 Важность признаков")
                    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
                    top_features = feature_importances.sort_values(ascending=False).head(10)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    top_features.plot(kind='barh', ax=ax, color='skyblue')
                    ax.set_title('Топ-10 важных признаков')
                    ax.set_xlabel('Важность')
                    st.pyplot(fig)
                    plt.close(fig)
                
            except Exception as e:
                st.error(f"Ошибка при обучении модели: {e}")
    
    # Информация о распределении классов
    st.sidebar.header("ℹ️ Информация о данных")
    fraud_percentage = (y.sum() / len(y)) * 100
    st.sidebar.metric("Мошеннических транзакций", 
                      f"{y.sum():,} ({fraud_percentage:.2f}%)")
    st.sidebar.metric("Законных транзакций", 
                      f"{(len(y) - y.sum()):,} ({(100 - fraud_percentage):.2f}%)")
    
    # Поддержка несбалансированных данных
    if fraud_percentage < 5:
        st.sidebar.warning("Внимание: данные сильно несбалансированы. Рекомендуется использовать балансировку классов.")

if __name__ == "__main__":
    main()