import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from tabulate import tabulate

# Carregar o dataset
arq_csv = pd.read_csv('datasets/iris.csv')
arq_csv = np.array(arq_csv)
dados = arq_csv[:, :-1].astype(np.float64)
classes = arq_csv[:, -1]

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
    pred_cross = cross_val_predict(classificador, dados, classes, cv=5)
    conf_matrix = confusion_matrix(classes, pred_cross)

    # Converter a matriz de confusão em uma string formatada
    conf_matrix_str = '\n'.join(' '.join(str(cell) for cell in row) for row in conf_matrix)

    resultados.append({
        "Classificador": nome,
        "Acurácia": f"{cv_resultados['test_accuracy'].mean():.5f}",
        "Precisão": f"{cv_resultados['test_precision_macro'].mean():.5f}",
        "Recall": f"{cv_resultados['test_recall_macro'].mean():.5f}",
        "F1-Score": f"{cv_resultados['test_f1_macro'].mean():.5f}",
        "Matriz de Confusão": conf_matrix_str
    })

# Converter a lista em um DataFrame
resultados_df = pd.DataFrame(resultados)

# Exibir o DataFrame com bordas usando a biblioteca tabulate
print(tabulate(resultados_df, headers='keys', tablefmt='grid', showindex=False))
