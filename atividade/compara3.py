import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, make_scorer, precision_recall_fscore_support, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Carregar o dataset
arq_csv = pd.read_csv('datasets/iris.csv')
arq_csv = np.array(arq_csv)
dados = arq_csv[:, :-1].astype(np.float64)
classes = arq_csv[:, -1]

# Dividir o dataset em treino e teste
# dados_treino, dados_teste, classes_treino, classes_teste = train_test_split(
#     dados, classes, test_size=0.2, random_state=42)

# Definir os classificadores
classificadores = {
    "Árvore de Decisão": DecisionTreeClassifier(criterion='entropy'),
    "SVM": SVC(kernel='poly'),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB()
}

# Definir as métricas
metricas = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

# Lista para armazenar os resultados
resultados = []

# Realizar a validação cruzada e calcular as métricas
for nome, classificador in classificadores.items():
    cv_resultados = cross_validate(classificador, dados, classes, cv=5, scoring=metricas)
    resultados.append({
        "Classificador": nome,
        "Acurácia": f"{cv_resultados['test_accuracy'].mean():.5f}",
        "Precisão": f"{cv_resultados['test_precision_macro'].mean():.5f}",
        "Recall": f"{cv_resultados['test_recall_macro'].mean():.5f}",
        "F1-Score": f"{cv_resultados['test_f1_macro'].mean():.5f}"
    })

# Converter a lista em um DataFrame
resultados_df = pd.DataFrame(resultados)

print(resultados_df)
