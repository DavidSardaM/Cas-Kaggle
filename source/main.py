import sklearn
from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
#matplotlib notebook
from matplotlib import pyplot as plt
import scipy.stats


# Funcio per a llegir dades en format csv
def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset

dataset = load_dataset('../data/dades.csv')

data = dataset.values

print(dataset.head())

#x = data[:, :64]
#y = data[:, 64]
print("Dimensionalitat de la BBDD:", dataset.shape)
#print("Dimensionalitat de les entrades X", x.shape)
#print("Dimensionalitat de l'atribut Y", y.shape)