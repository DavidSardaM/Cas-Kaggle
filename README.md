# Pràctica Kaggle APC UAB 2021-2022
### Nom: David Sardà Martin
### DATASET: Classify gestures by reading muscle activity.
### URL: https://www.kaggle.com/kyr7plus/emg-4

## Resum
El dataset utilitza dades preses per 8 diferents sensors i cada un dels sensors fa 8 lectures, amb aquestes 64 lectures intentarem predir 
els diferents gestos als quals corresponen les lectures. Les diferents lectures corresponen a 4 diferents gestos, aquests 4 gestos són:
pedra, tisores, paper i ok, aquestes classes estan codificades amb 0, 1, 2 i 3 respectivament.

Tenim un total 11.678 dades amb 65 atributs, els atributs corresponen a les 64 lectures dels 8 sensors amb 8 lectures diferents i finalment 
l'atribut que correspon a la classe que representa un gest. Totes les variables i atributs són 
numèriques i de tipus enter, ja que ja hi ha hagut la conversió a numèric del gest que representen. 

## Objectius del dataset

En aquest Kaggle volem predir amb la millor precisió possible les lectures dels diferents sensors per decidir a quin gest correspon. D'aquesta forma 
posteriorment per futures lectures podrà decidir a quin gest correspon i per exemple es podria automatitzar jugar el pedra, paper i tisores.

## Experiments

En aquesta pràctica hem provat diferents models de classificació, posteriorment per a millorar més el model buscarem els millors hiperparàmetres pel model 
i finalment obtindrem els millors models i hiperparàmetres segons la precisió i altres paràmetres per considerar les millors característiques. 


### Preprocessat

En el preprocessat hem normalitzat les dades prèviament a provar qualsevol model de classificació, hem vist que no tenim atributs Nans
per això no caldrà cap conversió dels atributs Nans. 

També s'han provat altres transformacions de les dades com el PCA utilitzant al mle, el PCA amb 3 components i el Polynomial Features de 
grau 2, amb aquestes transformacions no s'han aconseguit millores en la precisió de les dades en general, per la qual cosa es treballaran 
amb les dades sense transformacions.


### Model
| Model       | Hiperpàrametres                                   | Accuracy | f1_score | recall | roc     | Temps   |
| ----------- | ------------------------------------------------- | -------- | -------- | ------ | ------- | ------- |
| SVM   K=10   | C=1.0, kernel='rbf', gamma=0.9, probability=True  |  0.819747      |   0.819047        |  0.820289     |   0.978055    |   20.603432         |
| SVM   K=10   | C=1.0, kernel='rbf', gamma=0.7, probability=True  |  0.783354      |   0.782284        |  0.783918     |   0.971858    |    22.580357       |
| SVM   K=10   | C=1.0, kernel='sigmoid', gamma=0.9, probability=True | 0.244990       |    0.098365       |  0.2500     |   0.576449    |  48.316956          |
| SVM   K=10   | C=1.0, kernel='sigmoid', gamma=0.7, probability=True |  0.245334      |    0.098466       |  0.2500     |    0.566959   |  47.369840         |
| SVM   K=10   | C=1.0, kernel='poly', gamma=0.9, probability=True |   0.922333     |   0.922168        |    0.922634   |    0.989943   |    78.509772       |
| SVM   K=10   | C=1.0, kernel='poly', gamma=0.7, probability=True |   0.912056     |   0.911779        |    0.912379   |    0.990408   |    55.360835       |
| SVM   K=10   | C=1.0, kernel='linear', gamma=0.9, probability=True |   0.321289     |   0.326562     |    0.322482   |  0.548084     |    24.417258       |
| SVM   K=10  | C=1.0, kernel='linear', gamma=0.7, probability=True |   0.323258     |    0.327937     |    0.326656   |   0.546399    |    24.362318       |
| LogisticRegression   K=10   | C=2.0, fit_intercept=True, penalty='l2', max_iter=200000 |   0.345177     |    0.347963       |   0.345686    |   0.547031    |   3.231332         |
| Gaussian Naive Bayes   K=10  |     |  0.884484      |    0.883373       |   0.884478    |   0.977780    |    0.013022      |
| Linear Discriminant Analysis K=10  |   | 0.341067  |  0.347062  |  0.341717  |  0.548444  |  0.151041  |
| DecisionTree   K=10   |          |   0.787893        |   0.788465    |   0.788214    |   0.858743      |  0.518226   |
| KNeighbors  K=10   |          |     0.671261      |   0.636001    |   0.672889    |     0.871070      |      0.008373    |
| ExtraTrees  K=10  | n_estimators=100 |   0.942628    |   0.942448    |   0.942869    |     0.993144      |  1.446142   |
| Random Forest   K=10   | n_estimators=150, n_jobs=-1 |   0.920535     |     0.920134      |   0.920541    |   0.989419    |   1.922446   |
| HistGradientBoosting   K=10   | max_iter=100 |   0.961980     |    0.961891    |   0.961987    |   0.996782    |    2.767428   |
| AdaBoost   K=10   | n_estimators=150 |   0.875149     |    0.875251    |   0.875214    |  0.857691     |    3.671436    |
| Bagging   K=10   |  GaussianNB(), max_samples=0.9, max_features=0.9 |   0.884054   |   0.883074   |   0.884225    |   0.977215    |    0.101174    |
| Perceptron   K=10  | fit_intercept=False, max_iter=100, shuffle=True |        |           |       |       |           |
| GradientBoosting   K=10   | n_estimators=150 |     0.953332    |     0.953411      |    0.953614   |    0.995570   |    38.370672 |
| Classificador MLP   K=10   | solver='lbfgs',hidden_layer_sizes=(10,10,4, 4), max_iter=2000000 |    0.558242    |      0.513460   |    0.557942   |   0.789467    |   71.675886  |
| Random Forest (RS)    K=10   | bootstrap=False, criterion='entropy', max_depth=72, max_features='log2', n_estimators=460, n_jobs=-1, random_state=15 |    0.933809    |   0.933577  |   0.933943    |   0.992172     |    7.181002   |
| Random Forest (RS)   K=10   | bootstrap=False, criterion='entropy', max_depth=684, max_features='log2', n_estimators=270, n_jobs=-1, random_state=38 |  0.931667   |   0.931397  |    0.931649   |   0.991620    |   4.259888    |
| HistGradientBoosting (RS)   K=10  | learning_rate=0.3, max_depth=430, max_iter=357, max_leaf_nodes=26, random_state=69 |    0.961123   |   0.961152   |  0.961410  |   0.996867    |   1.171311  |
| ExtraTrees (RS)   K=10   | max_depth=500, n_estimators=397, n_jobs=-1, random_state=26 |  0.947166  |   0.946764   |  0.947129  |  0.993881  |  2.303823    |
| GradientBoosting (RS)  K=10   | criterion='squared_error', n_estimators=200, loss='deviance' , random_state=128 |  0.956414  |  0.956392   |   0.956521  |   0.996053    |  55.208113  |

Els darrers que contenen RS, s'han decidit els hiperparàmetres a partir d'un Random Search CV, per a poder triar els millors hiperparàmetres possibles.





## Demo

Podem provar d'executar una demo

## Conclusions

El millor model que s'ha aconseguit ha estat el HistGradientBoosting que ens ha aconseguit una accuracy del 96,%. Si ho comparem amb resultats obtinguts per altres desenvolupadors de Kaggles
que han intentat pel mateix dataset podem veure que, per exemple:

· Un cas per exemple és: https://www.kaggle.com/gauravduttakiit/hand-gesture-recognition-with-python per aquest cas aconsegueix un 92% d'accuracy, amb la qual cosa veiem que estem obtenint millors resultats.

· Un altre cas per comparar és: https://www.kaggle.com/gcdatkin/hand-gesture-prediction en aquest cas aconsegueix un 95% d'accuracy la qual cosa segueix essent lleugerament pitjor.

· També si ho comparem amb: https://www.kaggle.com/alincijov/muscles-ann-boosted-trees veiem que aconsegueix un 92% d'accuracy, inferior a aquest model.

Veient els resultats obtinguts podem veure que hem fet un bon model, una accuracy de predicció molt bona, també si veiem els altres paràmetres, podem veure de igual forma que aconseguim bons resultats.

## Idees per treballar en un futur

Si observem la matriu de confusió podem veure que les dades que més fallen són les de la classe 'ok' i 'paper' per això s'hauria de buscar la forma de poder diferenciar d'una millor forma aquestes dues classes. Es podria afegir un classificador que un cop obtinguts els resultats,
del nostre classificador intentes distingir especialment entre aquestes dues classes per aconseguir millors resultats.

## Llicencia
UAB