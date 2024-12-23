import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix


def main():
    iris_data = pd.read_csv('datasets\iris.csv')
    iris_data = np.array(iris_data)
    dados = iris_data[:, :-1]
    classes = iris_data[:, -1]

    svm_class = SVC(kernel='poly')
    mets = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']
    scores = cross_validate(svm_class, dados, classes, cv=5, scoring=mets)
    for s in scores:
        print("%s = Média %f Desvio padrão %f" % (s, np.average(scores[s]), np.std(scores[s])))

    preds = cross_val_predict(svm_class, dados, classes, cv=5)
    c_matrix = confusion_matrix(classes, preds)
    print(c_matrix)


if __name__ == '__main__':
    main()
