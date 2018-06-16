# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 20:57:43 2018

@author: takeshi
"""
import subprocess
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

def ejecutarPrograma(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f, feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to produce visualization")
   
# Read csv
df = pd.read_csv('pregunta_3.csv')
# Set index the column MouseID
df.set_index('Nro.')
# Get the X data encoding the string values as Genotype, Treatment, Behavior
X = df.iloc[:, :-1].apply(LabelEncoder().fit_transform)
x_values = X.values
# Get the output desired
y = df.iloc[:, 5]
y_values = y.values
   
# Create DT Classifier
dt = DecisionTreeClassifier(min_samples_split=2, random_state=1)
# Fit classifier
dt.fit(X, y)
ejecutarPrograma(dt, list(X.columns.values))

# Split training and test data
X_train, X_test, y_train, y_test = train_test_split(x_values, y_values, test_size=0.20)

# Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=2)
# Fit classifier
knn.fit(X_train, y_train)
# Predict and get accuracy score
print('KNN Accuracy score: ', accuracy_score(y_test, knn.predict(X_test)))

# Create Bayes Classifier
bayes = GaussianNB()
# Fit classifier
bayes.fit(X_train, y_train)
print('Bayes Accuracy score: ', accuracy_score(y_test, bayes.predict(X_test)))