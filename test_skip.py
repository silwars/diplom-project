import os
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import adjusted_rand_score, silhouette_score
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules

# путь к папке с файлами
data_folder = r"C:\Users\Гребенников Матвей\Desktop\Диплом\Курсовая\MachineLearningCSV\MachineLearningCVE"

# Список файлов
file_list = [os.path.join(data_folder, file) for file in os.listdir(data_folder) if file.endswith('.csv')]

# Объединяем файлы в один DataFrame
data = pd.concat([pd.read_csv(file) for file in file_list], ignore_index=True)

# Предобработка данных
# Предполагается, что последний столбец - это метки классов, а остальные - признаки
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Обработка NaN, Inf и слишком больших значений
X.replace([np.inf, -np.inf], np.nan, inplace=True)  # Заменяем Inf на NaN
X.fillna(X.mean(), inplace=True)  # Заменяем NaN средним значением по колонке
X = X.astype(np.float32)  # Приводим данные к типу float32, чтобы избежать ошибок переполнения

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Добавим возможность пропуска SVM и K-Means
skip_svm = input("Do you want to skip SVM algorithm? (yes/no): ").strip().lower() == 'yes'
skip_kmeans = input("Do you want to skip K-Means algorithm? (yes/no): ").strip().lower() == 'yes'

# Определяем алгоритмы для сравнения
algorithms = {
    'Decision Tree': DecisionTreeClassifier(),
    'Neural Network': MLPClassifier(max_iter=500),
    'Naive Bayes': GaussianNB(),
}

# добавляем SVM, если пользователь не выбрал пропуск
if not skip_svm:
    algorithms['Support Vector Machine'] = SVC()

# добавляем K-Means, если пользователь не выбрал пропуск
if not skip_kmeans:
    algorithms['K-Means (unsupervised)'] = KMeans(n_clusters=len(y.unique()), random_state=42)

# Оценка производительности алгоритмов
def evaluate_algorithms(X_train, X_test, y_train, y_test, algorithms):
    results = []
    for algo_name, algo in algorithms.items():
        print(f"\nEvaluating: {algo_name}")
        start_time = time.time()  # Начало отсчета времени
        if algo_name == 'K-Means (unsupervised)':
            # Для K-Means оцениваем качество кластеризации
            algo.fit(X_train)
            predictions = algo.predict(X_test)
            rand_index = adjusted_rand_score(y_test, predictions)
            silhouette = silhouette_score(X_test, predictions)
            elapsed_time = time.time() - start_time  # Конец отсчета времени
            print(f"  Adjusted Rand Index: {rand_index:.4f}")
            print(f"  Silhouette Score: {silhouette:.4f}")
            print(f"  Time Taken: {elapsed_time:.4f} seconds")
            results.append({
                'Algorithm': algo_name,
                'Metric': 'Adjusted Rand Index',
                'Value': rand_index,
                'Time (s)': elapsed_time
            })
            results.append({
                'Algorithm': algo_name,
                'Metric': 'Silhouette Score',
                'Value': silhouette,
                'Time (s)': elapsed_time
            })
        else:
            # Для других алгоритмов обучаем и тестируем
            algo.fit(X_train, y_train)
            predictions = algo.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            elapsed_time = time.time() - start_time  # Конец отсчета времени
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Time Taken: {elapsed_time:.4f} seconds")
            results.append({
                'Algorithm': algo_name,
                'Metric': 'Accuracy',
                'Value': accuracy,
                'Time (s)': elapsed_time
            })
    return pd.DataFrame(results)

# Оценка
results_df = evaluate_algorithms(X_train, X_test, y_train, y_test, algorithms)

# Выводим результаты
print("\nSummary of Results:")
print(results_df)

# Сохранение результатов
results_df.to_csv("algorithm_comparison_results_cicids2017.csv", index=False)

