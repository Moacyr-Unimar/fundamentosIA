import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix


def main():
    arq_csv = pd.read_csv('datasets/iris.csv')
    arq_csv = np.array(arq_csv)
    dados = arq_csv[:, :-1].astype(np.float64)
    classes = arq_csv[:, -1]

    nb = GaussianNB()
    d_tree = DecisionTreeClassifier(criterion='entropy')
    knn = KNeighborsClassifier(n_neighbors=5)
    svm_c = SVC(kernel='poly')

    mets = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']
    scores_naive = cross_validate(nb, dados, classes, cv=5, scoring=mets)
    scores_dtree = cross_validate(d_tree, dados, classes, cv=5, scoring=mets)
    scores_knn = cross_validate(knn, dados, classes, cv=5, scoring=mets)
    scores_svm = cross_validate(svm_c, dados, classes, cv=5, scoring=mets)

    print("Comparacao Árvore de Decisão, Naive Bayes, KNN e SVM")
    for s in scores_naive:
        print("%s ||| %.2f ||| %.2f ||| %.2f ||| %.2f ||| %.2f ||| %.2f ||| %.2f ||| %.2f "
              % (
                  s,
                  np.average(scores_dtree[s]),
                  np.std(scores_dtree[s]),
                  np.average(scores_naive[s]),
                  np.std(scores_naive[s]),
                  np.average(scores_knn[s]),
                  np.std(scores_knn[s]),
                  np.average(scores_svm[s]),
                  np.std(scores_svm[s])
              )
              )

    pred_cross = cross_val_predict(nb, dados, classes, cv=5)
    conf_matrix = confusion_matrix(classes, pred_cross)
    print("Matriz de confusão para o Naive Bayes")
    print(conf_matrix)

    pred_cross = cross_val_predict(d_tree, dados, classes, cv=5)
    conf_matrix = confusion_matrix(classes, pred_cross)
    print("Matriz de confusão para a Árvore de decisão")
    print(conf_matrix)

    pred_cross = cross_val_predict(knn, dados, classes, cv=5)
    conf_matrix = confusion_matrix(classes, pred_cross)
    print("Matriz de confusão para o KNN")
    print(conf_matrix)

    pred_cross = cross_val_predict(svm_c, dados, classes, cv=5)
    conf_matrix = confusion_matrix(classes, pred_cross)
    print("Matriz de confusão para o SVM")
    print(conf_matrix)


if __name__ == '__main__':
    main()
