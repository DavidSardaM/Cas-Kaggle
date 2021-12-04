import sklearn
from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
#matplotlib notebook
from matplotlib import pyplot as plt
import scipy.stats
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
import seaborn as sns
from sklearn.linear_model import Perceptron
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.neural_network import MLPClassifier

# Funcio per a llegir dades en format csv
def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset

dataset = load_dataset('../data/dades.csv')

data = dataset.values

print(dataset.head())

x = data[:, :64]
y = data[:, 64]
print("Dimensionalitat de la BBDD:", dataset.shape)
print("Dimensionalitat de les entrades X", x.shape)
print("Dimensionalitat de l'atribut Y", y.shape)

#pca
#polinomi


def standardize_mean(dataset):
    return MinMaxScaler().fit_transform(dataset)
    #return (dataset - dataset.mean(0)) / dataset.std(0)

x=standardize_mean(x)


models = []
#models.append(('SVM rbf gamma 0.9', SVC(C=1.0, kernel='rbf', gamma=0.9, probability=True)))
#models.append(('SVM rbf gamma 0.7', SVC(C=1.0, kernel='rbf', gamma=0.7, probability=True)))
#models.append(('SVM sigmoide gamma 0.9', SVC(C=1.0, kernel='sigmoid', gamma=0.9, probability=True)))
models.append(('SVM sigmoide gamma 0.7', SVC(C=1.0, kernel='sigmoid', gamma=0.7, probability=True)))
models.append(('SVM precomputed gamma 0.9', SVC(C=1.0, kernel='precomputed', gamma=0.9, probability=True)))
models.append(('SVM precomputed gamma 0.7', SVC(C=1.0, kernel='precomputed', gamma=0.7, probability=True)))
models.append(('SVM polinomi gamma 0.9', SVC(C=1.0, kernel='poly', gamma=0.9, probability=True)))
models.append(('SVM polinomi gamma 0.7', SVC(C=1.0, kernel='poly', gamma=0.7, probability=True)))
models.append(('SVM linear gamma 0.9', SVC(C=1.0, kernel='linear', gamma=0.9, probability=True)))
models.append(('SVM linear gamma 0.7', SVC(C=1.0, kernel='linear', gamma=0.7, probability=True)))
models.append (('Logistic Regression', LogisticRegression(C=2.0, fit_intercept=True, penalty='l2', tol=0.001)))
models.append (('Guassian Naive Bayes', GaussianNB()))
models.append (('Linear Discriminant Analysis', LinearDiscriminantAnalysis()))
models.append (('Decision Tree', DecisionTreeClassifier()))
models.append (('K Nearest Neigbors', KNeighborsClassifier()))
models.append (('Extra Trees', ExtraTreesClassifier(n_estimators=100)))
models.append (('Random Forest',  RandomForestClassifier( n_estimators=150, n_jobs=-1)))
models.append (('HistGradientBoosting', HistGradientBoostingClassifier(max_iter=100)))
models.append (('ADABoosting', AdaBoostClassifier(n_estimators=150)))
models.append (('Bagging Classifier', BaggingClassifier( GaussianNB(), max_samples=0.9, max_features=0.9)))
models.append (('Perceptró', Perceptron(fit_intercept=False, max_iter=100, shuffle=True)))
models.append (('GradientBoostingClassifier', GradientBoostingClassifier(n_estimators=150)))
models.append (('Red Neuronal MLPC', MLPClassifier(solver='lbfgs',hidden_layer_sizes=(10,10,4, 4))))











i_index=[2,3,4,6,10,20,40,60]

for index, (name, model) in enumerate(models):

        for i in i_index:
            K_Fold = model_selection.KFold (n_splits = i, shuffle=True)
            cv_results = model_selection.cross_val_score (model, x, y, cv = K_Fold, scoring = "accuracy")
            message =  "%s (%f):  %f  (%f)" % (name, i,cv_results.mean(), cv_results.std())
            print (message)


#pairplot
plt.figure()
sns.pairplot(dataset)
plt.savefig("../figures/histograma.png")
plt.show()


plt.figure()
sns.pairplot(dataset, hue="Forma")
plt.savefig("../figures/histograma_per_classes.png")
plt.show()