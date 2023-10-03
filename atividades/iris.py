# Importe as bibliotecas necessárias
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Carregue um conjunto de dados de exemplo (vamos usar o conjunto de dados Iris)
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Divida o conjunto de dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Padronize os recursos (isso é importante para SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crie um classificador SVM
svm_classifier = SVC(kernel='linear', C=1.0)

# Treine o classificador SVM
svm_classifier.fit(X_train, y_train)

# Faça previsões com o classificador treinado
y_pred = svm_classifier.predict(X_test)

# Avalie o desempenho do classificador
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Exiba a precisão e o relatório de classificação
print(f'Acurácia: {accuracy}')
print('Relatório de Classificação:\n', report)



