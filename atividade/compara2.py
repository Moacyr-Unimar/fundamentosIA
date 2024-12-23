import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


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

    classifiers = [nb, d_tree, knn, svm_c]
    classifier_names = ['Naive Bayes', 'Decision Tree', 'KNN', 'SVM']
    metrics = []

    for clf, name in zip(classifiers, classifier_names):
        pred_cross = cross_val_predict(clf, dados, classes, cv=5)
        precision = round(precision_score(classes, pred_cross, average='macro'), 5)
        recall = round(recall_score(classes, pred_cross, average='macro'), 5)
        f1 = round(f1_score(classes, pred_cross, average='macro'), 5)
        accuracy = round(np.average(cross_validate(clf, dados, classes, cv=5, scoring='accuracy')['test_score']), 5)
        metrics.append([name, precision, recall, f1, accuracy])

    df = pd.DataFrame(metrics, columns=['Classifier', 'Precision', 'Recall', 'F1-score', 'Accuracy'])

    # Encontrando os valores máximos
    max_precision = df['Precision'].max()
    max_recall = df['Recall'].max()
    max_f1 = df['F1-score'].max()
    max_accuracy = df['Accuracy'].max()

    # Substituindo os valores máximos por eles mesmos entre asteriscos duplos
    df['Precision'] = df['Precision'].apply(lambda x: f"**{round(x, 5)}**" if x == max_precision else round(x, 5))
    df['Recall'] = df['Recall'].apply(lambda x: f"**{round(x, 5)}**" if x == max_recall else round(x, 5))
    df['F1-score'] = df['F1-score'].apply(lambda x: f"**{round(x, 5)}**" if x == max_f1 else round(x, 5))
    df['Accuracy'] = df['Accuracy'].apply(lambda x: f"**{round(x, 5)}**" if x == max_accuracy else round(x, 5))

    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
