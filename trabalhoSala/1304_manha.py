import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, cross_validate
from sklearn.metrics import confusion_matrix


def breast_cancer_estimate_decision_tree():
    arq_csv = pd.read_csv('datasets/lungCancer.csv')
    dados = np.array(arq_csv)
    caracteristicas = dados[:, 3:-1].astype(np.float64)
    classes = dados[:, -1]

    d_tree = DecisionTreeClassifier(criterion='entropy')
    mets = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']
    scores = cross_validate(d_tree, caracteristicas, classes, cv=5, scoring=mets)
    print("Métricas ========================")
    for s in scores:
        print("Média da %s => %f === desvio padrão %f" % (s, np.average(scores[s]), np.std(scores[s])))

    pred_cross = cross_val_predict(d_tree, caracteristicas, classes, cv=10)
    conf_matrix = confusion_matrix(classes, pred_cross)
    print(conf_matrix)


def main():
    arq_csv = pd.read_csv('datasets/iris.csv')
    dados = np.array(arq_csv)
    caracteristicas = dados[:, :-1].astype(np.float64)
    classes = dados[:, -1]
    d_tree = DecisionTreeClassifier(criterion='entropy')

    mets = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']

    scores = cross_validate(d_tree, caracteristicas, classes, cv=10, scoring=mets)
    print("Métricas ========================")
    for s in scores:
        print("Média da %s => %f === desvio padrão %f" % (s, np.average(scores[s]), np.std(scores[s])))

    pred_cross = cross_val_predict(d_tree, caracteristicas, classes, cv=10)
    conf_matrix = confusion_matrix(classes, pred_cross)
    print(conf_matrix)
    # fTreino, fTeste, clasTreino, clasTeste = train_test_split(caracteristicas,classes)

    '''
    dTree.fit(fTreino,clasTreino)

    yPred = dTree.predict(fTeste)
    desempenho = [0,0]
    for i in range(len(yPred)):
        desempenho[int(yPred[i] == clasTeste[i])] +=1

    print("acertos: %d === erros %d" % (desempenho[1],desempenho[0]))
    '''


if __name__ == '__main__':
    breast_cancer_estimate_decision_tree()
