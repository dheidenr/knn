import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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

# Создание модели с указанием гиперпараметра C
clf = LogisticRegression(C=1)
clf.max_iter = 10000000
# Обучение модел
X_scaled = preprocessing.scale(train_feature_matrix)
clf.fit(X_scaled, train_labels)
# Предсказание на тестовой выборке
y_pred = clf.predict(test_feature_matrix)
print(y_pred)

print(accuracy_score(test_labels, y_pred))

clf = LogisticRegression(solver='saga')
# clf.max_iter = 1000
parad_grid = {
    'C': np.arange(1, 5),
    'penalty': ['l1', 'l2'],
}

search = GridSearchCV(
    clf, parad_grid, n_jobs=-1, cv=5, refit=True, scoring='accuracy')

search.fit(feature_matrix, labels)

# У меня с кучей варнингов {'C': 2, 'penalty': 'l2'} ghb max_
# На юпитере тоже в варниногом {'C': 1, 'penalty': 'l1'}

print(search.best_params_)


