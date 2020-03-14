import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

all_data = pd.read_csv('forest_dataset.csv')
print(all_data.head())

print(all_data.shape)
# Столбец '54' - метка класса
labels = all_data[all_data.columns[-1]].values

# Столбцы'0-54' - признаковые описания
feature_matrix = all_data[all_data.columns[:-1]].values

# Разделим выборку на обучающую и тестовую
train_feature_matrix, test_feature_matrix, train_labels, test_labels = \
    train_test_split(feature_matrix, labels, test_size=0.2, random_state=42)


clf = KNeighborsClassifier()

clf.fit(train_feature_matrix, train_labels)
y_pred = clf.predict(test_feature_matrix)
print(y_pred)

print(accuracy_score(test_labels, y_pred))
# При оптимальных параметрах 0.785
# По умолчанию 0.7365

clf = KNeighborsClassifier()
# clf.max_iter = 1000

parad_grid = {
    'n_neighbors': np.arange(1, 10),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan'],
}

search = GridSearchCV(clf, parad_grid, cv=5, scoring='accuracy', n_jobs=-1)

search.fit(feature_matrix, labels)

# У меня нашло с кучей варнингов {'C': 2, 'penalty': 'l2'} ghb max_
# Нашло на юпитере тоже в варниногом {'C': 1, 'penalty': 'l1'}

# {'metric': 'manhattan', 'n_neighbors': 4, 'weights': 'distance'}
print(search.best_params_)
# Вычисление оптимальной точности
clf = KNeighborsClassifier(
    n_neighbors=4, weights='distance', metric='manhattan')
clf.fit(train_feature_matrix, train_labels)
y_pred = clf.predict(test_feature_matrix)
print(y_pred)
print(accuracy_score(test_labels, y_pred))

# Вычисление прогназируемой вероятности класса под номером 3

optimal_clf = clf
pred_prob = clf.predict_proba(test_feature_matrix)
print(pred_prob)

unique, freq = np.unique(test_labels, return_counts=True)
freq = list(map(lambda x: x / len(test_labels), freq))

pred_freq = pred_prob.mean(axis=0)
plt.figure(figsize=(10, 8))
plt.bar(range(1, 8), pred_freq, width=0.4, align="edge", label='prediction')
plt.bar(range(1, 8), freq, width=-0.4, align="edge", label='real')
plt.ylim(0, 0.54)
plt.legend()
plt.show()
print(pred_freq[2])
