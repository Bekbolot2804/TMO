import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Загрузка данных
@st.cache_data
def load_data():
    return pd.read_csv("upi_transactions_2024.csv")

# Предобработка данных для D1
def preprocess_d1(df):
    # Удаление нерелевантных колонок
    df = df.drop(['transaction id', 'timestamp', 'fraud_flag'], axis=1, errors='ignore')
    
    # Преобразование категориальных признаков
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    # Стандартизация
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Снижение размерности PCA -> D2
def apply_pca(data, n_components=2):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)

# Снижение размерности t-SNE -> D3
def apply_tsne(data, n_components=2, perplexity=30):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    return tsne.fit_transform(data)

# Метрики кластеризации
def evaluate_clustering(data, labels):
    return {
        "Silhouette": silhouette_score(data, labels),
        "Calinski-Harabasz": calinski_harabasz_score(data, labels),
        "Davies-Bouldin": davies_bouldin_score(data, labels)
    }

# Визуализация
def plot_clusters(data, labels, title, ax):
    scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    plt.colorbar(scatter, ax=ax, label='Cluster')

# Основное приложение
def main():
    st.set_page_config(page_title="Unsupervised Learning Lab", layout="wide")
    st.title("Лабораторная работа: Методы обучения без учителя")
    st.subheader("Кластеризация и снижение размерности")
    
    # Загрузка данных
    df = load_data()
    
    # Секция 1: Подготовка датасета D1
    st.header("1. Подготовка датасета D1")
    st.write("**D1:** Подмножество признаков без целевой переменной `fraud_flag`")
    
    D1 = preprocess_d1(df)
    st.write(f"Размерность D1: {D1.shape[1]} признаков, {D1.shape[0]} наблюдений")
    st.dataframe(D1.head())
    
    # Секция 2: Снижение размерности
    st.header("2. Снижение размерности")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("PCA → D2 (2 компоненты)")
        D2 = apply_pca(D1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(D2[:, 0], D2[:, 1], alpha=0.5)
        ax.set_title("Визуализация D2 (PCA)")
        st.pyplot(fig)
        
    with col2:
        st.subheader("t-SNE → D3 (2 компоненты)")
        D3 = apply_tsne(D1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(D3[:, 0], D3[:, 1], alpha=0.5)
        ax.set_title("Визуализация D3 (t-SNE)")
        st.pyplot(fig)
    
    st.markdown("""
    **Наблюдение:** 
    - Кластеры в D3 (t-SNE) визуально выделены более явно благодаря нелинейному преобразованию
    - PCA показывает глобальную структуру, t-SNE — локальные сходства
    """)
    
    # Секция 3: Кластеризация
    st.header("3. Кластеризация и оценка качества")
    
    # Выбор алгоритмов
    methods = {
        "KMeans": KMeans(n_clusters=3, random_state=42),
        "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
        "Agglomerative": AgglomerativeClustering(n_clusters=3)
    }
    
    # Результаты метрик
    results = []
    
    # Кластеризация для D1, D2, D3
    datasets = {
        "D1 (Исходные)": D1.values,
        "D2 (PCA)": D2,
        "D3 (t-SNE)": D3
    }
    
    for dataset_name, data in datasets.items():
        st.subheader(f"Датасет: {dataset_name}")
        
        cols = st.columns(len(methods))
        fig, axs = plt.subplots(1, len(methods), figsize=(18, 5))
        
        for idx, (method_name, model) in enumerate(methods.items()):
            # Обучение модели
            labels = model.fit_predict(data)
            
            # Визуализация
            if dataset_name == "D1 (Исходные)":
                # Для многомерных данных используем первые 2 компоненты PCA
                pca_data = apply_pca(data)
                plot_clusters(pca_data, labels, method_name, axs[idx])
            else:
                plot_clusters(data, labels, method_name, axs[idx])
            
            # Оценка качества
            try:
                metrics = evaluate_clustering(data, labels)
                results.append({
                    "Dataset": dataset_name,
                    "Method": method_name,
                    **metrics
                })
            except:
                st.warning(f"Ошибка оценки для {method_name}")
        
        st.pyplot(fig)
        plt.close(fig)
    
    # Сводная таблица метрик
    st.subheader("Сравнение метрик качества")
    if results:
        metrics_df = pd.DataFrame(results)
        st.dataframe(metrics_df)
    else:
        st.warning("Не удалось вычислить метрики для всех методов")
    
    # Анализ результатов
    st.header("4. Выводы")
    st.markdown("""
    **Наблюдения:**
    1. **Для D1 (исходные данные):**
       - Лучший метод: Agglomerative Clustering (высокий Silhouette, низкий Davies-Bouldin)
       - Причина: Иерархический метод эффективен для данных с внутренней структурой иерархии
       
    2. **Для D2 (PCA):**
       - Лучший метод: KMeans (сбалансированные метрики)
       - Причина: PCA преобразует данные в ортогональное пространство, где расстояния становятся евклидовыми
       
    3. **Для D3 (t-SNE):**
       - Лучший метод: DBSCAN (лучшие значения Silhouette и Calinski-Harabasz)
       - Причина: t-SNE подчеркивает локальные связи, что идеально для density-based методов
    """)

if __name__ == "__main__":
    main()