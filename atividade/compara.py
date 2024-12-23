import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score
from tabulate import tabulate


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
        data = [
            ['Decision Tree', np.average(scores_dtree[s]), np.std(scores_dtree[s])],
            ['Naive Bayes', np.average(scores_naive[s]), np.std(scores_naive[s])],
            ['KNN', np.average(scores_knn[s]), np.std(scores_knn[s])],
            ['SVM', np.average(scores_svm[s]), np.std(scores_svm[s])]
        ]
        print(tabulate(data, headers=['Classifier', 'Mean', 'Std'], tablefmt='pretty'))

    # Analisando os resultados
    classifiers = [nb, d_tree, knn, svm_c]
    classifier_names = ['Naive Bayes', 'Decision Tree', 'KNN', 'SVM']
    accuracies = []

    for clf, name in zip(classifiers, classifier_names):
        pred_cross = cross_val_predict(clf, dados, classes, cv=5)
        acc = accuracy_score(classes, pred_cross)
        accuracies.append(acc)
        conf_matrix = confusion_matrix(classes, pred_cross)
        print(f"Matriz de confusão para o {name}")
        print(conf_matrix)

    max_acc_index = accuracies.index(max(accuracies))
    print(f"\nO melhor classificador para este conjunto de dados é: {
          classifier_names[max_acc_index]} com uma precisão de {accuracies[max_acc_index]}")


if __name__ == '__main__':
    main()
