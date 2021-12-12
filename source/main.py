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
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score, roc_curve, auc, classification_report, accuracy_score, make_scorer
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.linear_model import Perceptron
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split


# Funcio per a llegir dades en format csv
def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset

dataset = load_dataset('../data/dades.csv')



def analisi_dades(dataset):


    print("Primer observem les dades y estadístiques sobre les dades")
    print(dataset.head())
    print(dataset.describe())
    data = dataset.values
    x = data[:, :64]
    y = data[:, 64]
    print("Seleccionem la variable objectiu i mirarem les dimensionalitats de les nostres dades")

    print("Dimensionalitat de la BBDD:", dataset.shape)
    print("Dimensionalitat de les entrades X", x.shape)
    print("Dimensionalitat de l'atribut Y", y.shape)


    print("Posteriorment mirarem el tipus de dades que tenim")
    print(dataset.dtypes)

    print("Aixó com també observarem el nombre de dades nules que tenim")
    print(dataset.isnull().sum())

    return data, x, y

data, x, y=analisi_dades(dataset)



def distribucions(dataset):
    print("Un cop començades a veure les dades passarem a observar les distribucions i relacions que considerem interessants")
    print("Primer observarem la matriu de correlació dels atributs")
    plt.figure()
    fig, ax = plt.subplots(figsize=(25, 10))  # figsize controla l'amplada i alçada de les cel·les de la matriu
    plt.title("Matriu de correlació de Pearson")
    sns.heatmap(dataset.corr(), annot=True, ax=ax, linewidths=.0, annot_kws={"fontsize": 350 / np.sqrt(len(dataset))},
                square=True)
    plt.savefig("../figures/pearson_correlation_matrix_.png")
    plt.show()
    print("Posteriorment passarem a veure els histogrames de les variables")
    print("Primer els histogrames de tots els atributs")
    #plt.figure()
    #sns.pairplot(dataset)
    #plt.savefig("../figures/histograma.png")
    #plt.show()



    print("Després els histogrames dels atributs segons la classe objectiu")

    #plt.figure()
    #sns.pairplot(dataset, hue="Forma", palette={0: 'thistle', 1: "lightskyblue",  2: "lightcoral", 3: "lightgreen"})
    #plt.savefig("../figures/histograma_per_classes.png")
    plt.show()

    print("Finalment veurem la distribució de la variables objectiu i si les classes es troben balancejades, si és el cas la precisió de les dades serà molt més reprsentativa de les dades")
    plt.figure()
    ax = sns.countplot(x="Forma", data=dataset, palette={0: 'thistle', 1: "lightskyblue", 2: "lightcoral", 3: "lightgreen"})
    plt.suptitle("Target attribute distribution (Forma)")
    label = ["rock", "scissors", "paper", "ok"]
    ax.bar_label(container=ax.containers[0], labels=label)
    plt.xlabel('Forma')
    plt.ylabel('Number of samples')
    plt.savefig("../figures/distribucio_atribut_objectiu.png")
    plt.show()

    porc_pot = (len(dataset[dataset.Forma == 0]) / len(dataset.Forma)) * 100
    print('El percentatge de gestos de les mans que son pedres representa un {:.2f}% del total de dades'.format(porc_pot))
    porc_pot = (len(dataset[dataset.Forma == 1]) / len(dataset.Forma)) * 100
    print('El percentatge de gestos de les mans que son tisores representa un {:.2f}% del total de dades'.format(porc_pot))
    porc_pot = (len(dataset[dataset.Forma == 2]) / len(dataset.Forma)) * 100
    print('El percentatge de gestos de les mans que son papers representa un {:.2f}% del total de dades'.format(porc_pot))
    porc_pot = (len(dataset[dataset.Forma == 3]) / len(dataset.Forma)) * 100
    print('El percentatge de gestos de les mans que son ok representa un {:.2f}% del total de dades'.format(porc_pot))


#distribucions(dataset)

# +--------------------------+
# | TRANSFORMACIONS          |
# +--------------------------+
#Posteriorment veurem possibles transformacions de les dades que poden ajudar a millorar el model de classificació com son: el pca y el polinomial Features

#pca
def Pca(x, num):
    pca = PCA(n_components=num)
    pca.fit(x)
    x_pca=pca.transform(x)
    return x_pca

x_pca1=Pca(x,3)

x_pca2=Pca(x,'mle')
#polinomi

def polinomi(x, num):
    trans = PolynomialFeatures(degree=num)
    data = trans.fit_transform(x)
    return data

x_poli=polinomi(x, 2)

#Per treballar amb les dades és adequat estandaritzar-los, i en el nostre cas com les variables tenen distribucions Gaussianes ens donarà millors resultats.
def standardize_mean(dataset):
    return MinMaxScaler().fit_transform(dataset)


#x=standardize_mean(x)


# +--------------------------+
# | PROVA DE MODELS          |
# +--------------------------+
#A continuació provarem una sèrie de models per veure la millor opció


models = []
#models.append(('SVM rbf gamma 0.9', make_pipeline(MinMaxScaler(), SVC(C=1.0, kernel='rbf', gamma=0.9, probability=True))))
#models.append(('SVM rbf gamma 0.7', make_pipeline(MinMaxScaler(), SVC(C=1.0, kernel='rbf', gamma=0.7, probability=True))))
#models.append(('SVM sigmoide gamma 0.9', make_pipeline(MinMaxScaler(), SVC(C=1.0, kernel='sigmoid', gamma=0.9, probability=True))))
#models.append(('SVM sigmoide gamma 0.7', make_pipeline(MinMaxScaler(), SVC(C=1.0, kernel='sigmoid', gamma=0.7, probability=True))))
#models.append(('SVM precomputed gamma 0.9', make_pipeline(MinMaxScaler(), SVC(C=1.0, kernel='precomputed', gamma=0.9, probability=True))))
#models.
# 2wsz 3('SVM precomputed gamma 0.7', make_pipeline(MinMaxScaler(),SVC(C=1.0, kernel='precomputed', gamma=0.7, probability=True))))
#models.append(('SVM polinomi gamma 0.9', make_pipeline(MinMaxScaler(),SVC(C=1.0, kernel='poly', gamma=0.9, probability=True))))
#models.append(('SVM polinomi gamma 0.7', make_pipeline(MinMaxScaler(),SVC(C=1.0, kernel='poly', gamma=0.7, probability=True))))
#models.append(('SVM linear gamma 0.9', make_pipeline(MinMaxScaler(),SVC(C=1.0, kernel='linear', gamma=0.9, probability=True))))
#models.append(('SVM linear gamma 0.7', make_pipeline(MinMaxScaler(),SVC(C=1.0, kernel='linear', gamma=0.7, probability=True))))
#models.append (('Logistic Regression', make_pipeline(MinMaxScaler(),LogisticRegression(C=2.0, fit_intercept=True, penalty='l2', max_iter=200000))))
#models.append (('Guassian Naive Bayes', make_pipeline(MinMaxScaler(),GaussianNB())))
#models.append (('Linear Discriminant Analysis', make_pipeline(MinMaxScaler(),LinearDiscriminantAnalysis())))
#models.append (('Decision Tree', make_pipeline(MinMaxScaler(),DecisionTreeClassifier())))
#models.append (('K Nearest Neigbors', make_pipeline(MinMaxScaler(),KNeighborsClassifier())))
#models.append (('Extra Trees', make_pipeline(MinMaxScaler(),ExtraTreesClassifier(n_estimators=100))))
#models.append (('Random Forest',  make_pipeline(MinMaxScaler(),RandomForestClassifier( n_estimators=150, n_jobs=-1))))
#models.append (('HistGradientBoosting', make_pipeline(MinMaxScaler(),HistGradientBoostingClassifier(max_iter=100))))
#models.append (('ADABoosting', make_pipeline(MinMaxScaler(),AdaBoostClassifier(n_estimators=150))))
#models.append (('Bagging Classifier', make_pipeline(MinMaxScaler(),BaggingClassifier( GaussianNB(), max_samples=0.9, max_features=0.9))))
#models.append (('Perceptró', make_pipeline(MinMaxScaler(),Perceptron(fit_intercept=False, max_iter=100, shuffle=True))))
#models.append (('GradientBoostingClassifier', make_pipeline(MinMaxScaler(),GradientBoostingClassifier(n_estimators=50))))
#models.append (('Red Neuronal MLPC', make_pipeline(MinMaxScaler(),MLPClassifier(solver='lbfgs',hidden_layer_sizes=(10,10,4, 4), max_iter=2000000))))

#models.append (('Optimized Random Forest 1', make_pipeline(MinMaxScaler(),RandomForestClassifier(bootstrap=False, criterion='entropy', max_depth=72,
                       #max_features='log2', n_estimators=460, n_jobs=-1, random_state=15))))
#models.append (('Optimized Random Forest 2', make_pipeline(MinMaxScaler(),RandomForestClassifier(bootstrap=False, criterion='entropy', max_depth=684,
                       #max_features='log2', n_estimators=270, n_jobs=-1, random_state=38))))
#models.append (('Optimized Hist Gradient Boosting', make_pipeline(MinMaxScaler(),HistGradientBoostingClassifier(learning_rate=0.3, max_depth=430, max_iter=357,
                              # max_leaf_nodes=26, random_state=69))))
#models.append (('Optimized ExtraTrees', make_pipeline(MinMaxScaler(),ExtraTreesClassifier(max_depth=500, n_estimators=397, n_jobs=-1,
                     #random_state=26))))
#models.append (('Optimized Gradient Boosting', make_pipeline(MinMaxScaler(),GradientBoostingClassifier(criterion='squared_error', n_estimators=200, loss='deviance' , random_state=128) )))








i_index=[2,3,4,5,6,10]
scoring = ['accuracy','f1_macro',  'recall_macro',  'roc_auc_ovr']



for index, (name, model) in enumerate(models):

        for i in i_index:
            K_Fold = model_selection.KFold (n_splits = i, shuffle=True)
            cv_results = model_selection.cross_validate (model, x_poli, y, cv = K_Fold, scoring = scoring)
            message =  "%s  dades polinomi 2 (%f):  accuracy: %f (%f),  f1: %f, recall: %f, roc: %f tiempo %f " % (name, i,cv_results['test_accuracy'].mean(),
                                    cv_results['test_accuracy'].std(),  cv_results['test_f1_macro'].mean(), cv_results['test_recall_macro'].mean(),
                                    cv_results['test_roc_auc_ovr'].mean(), cv_results['fit_time'].mean() )
            print (message)

# +--------------------------+
# |   CERCA HIPERPARAMETRES  |
# +--------------------------+

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


# +--------------------------+
# |   MODEL ELEGIT            |
# +--------------------------+


n_classes=4



Xtrain, Xtest, ytrain, ytest = train_test_split(x, y)
clf=HistGradientBoostingClassifier( learning_rate=0.3, max_depth=430, max_iter=357, max_leaf_nodes=26, random_state=69)
clf.fit(Xtrain, ytrain)
print(clf.score(Xtest,ytest))



probs = clf.predict_proba(Xtest)
predict=clf.predict(Xtest)


# +--------------------------+
# |   ANALISI RESULTATS      |
# +--------------------------+

cf_m=confusion_matrix(ytest, predict)
plt.figure()
sns.heatmap(cf_m, annot=True, cmap="YlGnBu")

plt.savefig("../figures/confusion_matrix.png")
plt.show()



# Corba precision recall
precision = {}
recall = {}
average_precision = {}
plt.figure()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(ytest == i, probs[:, i])
    average_precision[i] = average_precision_score(ytest == i, probs[:, i])

    plt.plot(recall[i], precision[i],
             label='Precision-recall curve of class {0} (area = {1:0.2f})'
                   ''.format(i, round(average_precision[i])))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="upper right")
plt.savefig("../figures/corba_precision_recall.png")
plt.show()
#corba ROC
fpr = {}
tpr = {}
roc_auc = {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(ytest == i, probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


plt.figure()
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})' ''.format(i, roc_auc[i]))
plt.legend()
plt.savefig("../figures/corba_roc.png")
plt.show()



