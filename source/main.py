import sklearn
from sklearn.datasets import make_regression

import pandas as pd
#matplotlib notebook

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
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures

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

def print_data_types(dataset):
    print("------------------------------------")
    print("Tipus de dades")
    print(dataset.dtypes)
    print("------------------------------------")

print_data_types(dataset)

def nan(dataset):
    print("Eliminació 'NaN' del DataFrame")
    print(dataset.isnull().sum())
    print("------------------------------------")

nan(dataset)

def balance(dataset):
    ax = sns.countplot(x="Forma", data=dataset, palette={0: 'thistle', 1: "lightskyblue",  2: "lightcoral", 3: "lightgreen"})
    plt.suptitle("Target attribute distribution (Forma)")
    label = ["rock",  "scissors","paper", "ok"]
    ax.bar_label(container=ax.containers[0], labels=label)
    plt.xlabel('Forma')
    plt.ylabel('Number of samples')
    plt.savefig("../figures/distribucio_atribut_objectiu.png")
    plt.show()

    porc_pot = (len(dataset[dataset.Forma == 0]) / len(dataset.Forma)) * 100
    print('The samples that are rock is: {:.2f}%'.format(porc_pot))
    porc_pot = (len(dataset[dataset.Forma == 1]) / len(dataset.Forma)) * 100
    print('The samples that are scissors is: {:.2f}%'.format(porc_pot))
    porc_pot = (len(dataset[dataset.Forma == 2]) / len(dataset.Forma)) * 100
    print('The samples that are paper is: {:.2f}%'.format(porc_pot))
    porc_pot = (len(dataset[dataset.Forma == 3]) / len(dataset.Forma)) * 100
    print('The samples that are rock is: {:.2f}%'.format(porc_pot))
#balance(dataset)

def pearson_correlation(dataset):
    plt.figure()
    fig, ax = plt.subplots(figsize=(25, 10))  # figsize controla l'amplada i alçada de les cel·les de la matriu
    plt.title("Matriu de correlació de Pearson")
    sns.heatmap(dataset.corr(), annot=True, ax=ax, linewidths=.0, annot_kws={"fontsize":350 / np.sqrt(len(dataset))},square=True)
    plt.savefig("../figures/pearson_correlation_matrix_.png")
    plt.show()

#pearson_correlation(dataset)
#pca
def Pca(x):
    pca = PCA(n_components=3)
    pca.fit(x)
    x_pca=pca.transform(x)
    return x_pca

#x_pca=Pca(x)
#polinomi

def polinomi(x):
    trans = PolynomialFeatures(degree=2)
    data = trans.fit_transform(x)
    return data

#x_poli=polinomi(x)
def standardize_mean(dataset):
    return MinMaxScaler().fit_transform(dataset)
    #return (dataset - dataset.mean(0)) / dataset.std(0)

x=standardize_mean(x)


models = []
#models.append(('SVM rbf gamma 0.9', SVC(C=1.0, kernel='rbf', gamma=0.9, probability=True)))
#models.append(('SVM rbf gamma 0.7', SVC(C=1.0, kernel='rbf', gamma=0.7, probability=True)))
#models.append(('SVM sigmoide gamma 0.9', SVC(C=1.0, kernel='sigmoid', gamma=0.9, probability=True)))
#models.append(('SVM sigmoide gamma 0.7', SVC(C=1.0, kernel='sigmoid', gamma=0.7, probability=True)))
#models.append(('SVM precomputed gamma 0.9', SVC(C=1.0, kernel='precomputed', gamma=0.9, probability=True)))
#models.append(('SVM precomputed gamma 0.7', SVC(C=1.0, kernel='precomputed', gamma=0.7, probability=True)))
#models.append(('SVM polinomi gamma 0.9', SVC(C=1.0, kernel='poly', gamma=0.9, probability=True)))
#models.append(('SVM polinomi gamma 0.7', SVC(C=1.0, kernel='poly', gamma=0.7, probability=True)))
#models.append(('SVM linear gamma 0.9', SVC(C=1.0, kernel='linear', gamma=0.9, probability=True)))
#models.append(('SVM linear gamma 0.7', SVC(C=1.0, kernel='linear', gamma=0.7, probability=True)))
#models.append (('Logistic Regression', LogisticRegression(C=2.0, fit_intercept=True, penalty='l2', max_iter=200000)))
#models.append (('Guassian Naive Bayes', GaussianNB()))
#models.append (('Linear Discriminant Analysis', LinearDiscriminantAnalysis()))
#models.append (('Decision Tree', DecisionTreeClassifier()))
#models.append (('K Nearest Neigbors', KNeighborsClassifier()))
#models.append (('Extra Trees', ExtraTreesClassifier(n_estimators=100)))
#models.append (('Random Forest',  RandomForestClassifier( n_estimators=150, n_jobs=-1)))
#models.append (('HistGradientBoosting', HistGradientBoostingClassifier(max_iter=100)))
#models.append (('ADABoosting', AdaBoostClassifier(n_estimators=150)))
#models.append (('Bagging Classifier', BaggingClassifier( GaussianNB(), max_samples=0.9, max_features=0.9)))
#models.append (('Perceptró', Perceptron(fit_intercept=False, max_iter=100, shuffle=True)))
#models.append (('GradientBoostingClassifier', GradientBoostingClassifier(n_estimators=150)))
#models.append (('Red Neuronal MLPC', MLPClassifier(solver='lbfgs',hidden_layer_sizes=(10,10,4, 4), max_iter=2000000)))

#models.append (('Optimized Random Forest 1', RandomForestClassifier(bootstrap=False, criterion='entropy', max_depth=72,
                      # max_features='log2', n_estimators=460, n_jobs=-1, random_state=15)))
#models.append (('Optimized Random Forest 2',RandomForestClassifier(bootstrap=False, criterion='entropy', max_depth=684,
                      # max_features='log2', n_estimators=270, n_jobs=-1, random_state=38)))
#models.append (('Optimized Hist Gradient Boosting',HistGradientBoostingClassifier(learning_rate=0.3, max_depth=430, max_iter=357,
                      #         max_leaf_nodes=26, random_state=69)))
#models.append (('Optimized ExtraTrees',ExtraTreesClassifier(max_depth=500, n_estimators=397, n_jobs=-1,
                     #random_state=26)))
#models.append (('Optimized Gradient Boosting',GradientBoostingClassifier(criterion='squared_error', n_estimators=200, loss='deviance' , random_state=128) ))








i_index=[2,3,4,6,10,20,40,60]

for index, (name, model) in enumerate(models):

        for i in i_index:
            K_Fold = model_selection.KFold (n_splits = i, shuffle=True)
            cv_results = model_selection.cross_val_score (model, x, y, cv = K_Fold, scoring = "accuracy")
            message =  "%s (%f):  %f  (%f)" % (name, i,cv_results.mean(), cv_results.std())
            print (message)


#pairplot
#plt.figure()
#sns.pairplot(dataset)
#plt.savefig("../figures/histograma.png")
#plt.show()


#plt.figure()
#sns.pairplot(dataset, hue="Forma", palette={0: 'thistle', 1: "lightskyblue",  2: "lightcoral", 3: "lightgreen"})
#plt.savefig("../figures/histograma_per_classes.png")
#plt.show()

"""
param = {
    'bootstrap': [True, False],
    'max_depth': np.arange(4,1000),
    'min_samples_leaf': [1, 3, 4, 5],
    'min_samples_split': [2, 8, 10, 12],
    'n_estimators': np.arange(50,500),
    'criterion': ['gini', 'entropy'],
    'max_features': ['auto', 'sqrt', 'log2'],
    'random_state': np.arange(0,1000)

}

rf = RandomForestClassifier(n_jobs=-1)# Instantiate the grid search model
random_search_random = RandomizedSearchCV(estimator = rf, param_distributions = param, cv = 3, n_jobs = -1, n_iter=500)
random_search_random = random_search_random.fit(x, y)
best_estimator = random_search_random.best_estimator_
print(best_estimator)


et = ExtraTreesClassifier(n_jobs=-1)# Instantiate the grid search model
random_search_extra = RandomizedSearchCV(estimator = et, param_distributions = param, cv = 3, n_jobs = -1, n_iter=500)
random_search_extra = random_search_extra.fit(x, y)
best_estimator1 = random_search_extra.best_estimator_
print(best_estimator1)

param = {
    'max_depth': np.arange(4,1000),
    'max_leaf_nodes': np.arange(2,500),
    'max_iter': np.arange(50,500),
    'loss' :['auto', 'categorical_crossentropy'],
    'random_state': np.arange(0,1000),
    'learning_rate': [0.00001,0.0001,0.001,0.01,0.05,0.1,0.3,0.5, 0.8, 1]

}

hgb = HistGradientBoostingClassifier()# Instantiate the grid search model
random_search_hist = RandomizedSearchCV(estimator = hgb, param_distributions = param, cv = 3, n_jobs = -1, n_iter=500)
random_search_hist = random_search_hist.fit(x, y)
best_estimator = random_search_hist.best_estimator_
print(best_estimator)


param = {
    'max_depth': np.arange(2,1000),
    'max_leaf_nodes': np.arange(2,500),
    'n_estimators': np.arange(50,500),
    'loss': ['deviance'],
    'random_state': np.arange(0,1000),
    'criterion': ['friedman_mse', 'squared_error',  'absolute_error'],
    'min_samples_split': [2, 8, 10, 12],



}

gb = GradientBoostingClassifier()# Instantiate the grid search model
random_search_gradient = RandomizedSearchCV(estimator = gb, param_distributions = param, cv = 3, n_jobs = -1, n_iter=10)
random_search_gradient = random_search_gradient.fit(x, y)
best_estimator2 = random_search_gradient.best_estimator_
print(best_estimator2)

"""
