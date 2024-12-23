import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.metrics import confusion_matrix


def main():
    arq_csv = pd.read_csv('datasets\iris.csv')
    arq_csv = np.array(arq_csv)
    dados_treino_teste = arq_csv[:, :-1].astype(np.float64)
    classes = arq_csv[:, -1]

    knn = KNeighborsClassifier(n_neighbors=3)
    mets = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']
    scores = cross_validate(knn, dados_treino_teste, classes, cv=5, scoring=mets)
    for s in scores:
        print("%s ==> Média -> %f | Desvio padrão -> %f" % (
            s, np.average(scores[s]), np.std(scores[s])))

    pred = cross_val_predict(knn, dados_treino_teste, classes, cv=5)
    conf_matrix = confusion_matrix(classes, pred)
    print(conf_matrix)


if __name__ == "__main__":
    main()
