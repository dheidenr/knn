import pandas as pd
import numpy as np

all_data = pd.read_csv('forest_dataset.csv')
print(all_data.head())

print(all_data.shape)
# Столбец '54' - метка класса
labels = all_data[all_data.columns[-1]].values

# Столбцы'0-54' - признаковые описания
feature_matrix = all_data[all_data.columns[:-1]].values
