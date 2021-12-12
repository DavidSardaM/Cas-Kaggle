import pandas as pd

import sys
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from random import randrange
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset


dataset = load_dataset('../data/dades.csv')

if len(sys.argv)==1:
    predir = pd.DataFrame(columns=dataset.columns)
    for i in range(0,100):
        x= randrange(len(dataset))
        predir.loc[i] = dataset.iloc[x,: ]
else:
    predir = sys.argv[1]



def standardize_mean(dataset):
    return MinMaxScaler().fit_transform(dataset)




data = dataset.values
x = data[:, :64]
y = data[:, 64]
x=standardize_mean(x)

dre = predir.values
x_pred = dre[:, :64]
x_pred=standardize_mean(x_pred)
y_pred = dre[:, 64]


clf = HistGradientBoostingClassifier( learning_rate=0.3, max_depth=430, max_iter=357, max_leaf_nodes=26, random_state=69 )
clf.fit(x, y)

predides2 = clf.predict(x_pred)


