import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def load_dataset(path_dataset):
    dataset_csv = pd.read_csv(path_dataset)
    data = np.array(dataset_csv)
    return data[:, :-1].astype(np.float64), data[:, -1], dataset_csv.columns


def main():
    data, classes, _ = load_dataset('datasets/iris.csv')
    le = preprocessing.LabelEncoder()
    yvs = le.fit_transform(classes)

    x_train, x_test, y_train, y_test = train_test_split(data, yvs, random_state=0)

    de_class = GaussianNB()
    de_class.fit(x_train, y_train)
    guessed = de_class.predict(x_test)
    print(classification_report(y_test, guessed))


if __name__ == '__main__':
    main()
