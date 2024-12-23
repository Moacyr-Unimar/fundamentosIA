import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from tabulate import tabulate
import matplotlib.pyplot as plt

# Carregar o dataset
arq_csv = pd.read_csv('datasets/ecg/mitbih_train.csv', header=None)

# Assuming the target column is the last column
target_column_index = arq_csv.shape[1] - 1

# Get the unique values in the target column
class_labels = arq_csv[target_column_index].unique()

# Assign meaningful names to the class labels based on domain knowledge
class_names = {
    0: "Normal Beats",
    1: "Supraventricular Ectopy Beats",
    2: "Ventricular Ectopy Beats",
    3: "Fusion Beats",
    4: "Unclassifiable Beats"
}

# Print class labels with their assigned names
for label in class_labels:
    print(f"Class label {label}: {class_names[label]}")

# Explore dataset
print("Columns and their NaN percentages:")
null_col = arq_csv.isna().mean() * 100
print(null_col)

arq_csv[187] = arq_csv[187].astype(float)
equilibre = arq_csv[187].value_counts()
print(equilibre)

arq_csv.info()


# Get a pie chart that explain every class with its perecentages in the training dataset

plt.figure(figsize=(20, 10))
my_circle = plt.Circle((0, 0), 0.7, color='white')
plt.pie(equilibre, labels=['Normal Beats', 'Supraventricular Ectopy Beats', 'Ventricular Ectopy Beats', 'Fusion Beats', 'Unclassifiable Beats'], colors=[
        'Blue', 'Green', 'Yellow', 'Skyblue', 'Orange'], autopct='%1.1f%%', textprops={'color': 'black'})
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.show()
