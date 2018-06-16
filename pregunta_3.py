from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Read csv
df = pd.read_csv('pregunta_3.csv')
# Set index the column Nro
df.set_index('Nro')
# Get the X data encoding the string values as Genotype, Treatment, Behavior
X = df.drop('Nro', 1).iloc[:, :-1].apply(LabelEncoder().fit_transform)
x_values = X.values
# Get the output desired
y = df.iloc[:, 5]
y_values = y.values

dtree=DecisionTreeClassifier()
dtree.fit(x_values,y_values)

dot_data = StringIO()

export_graphviz(dtree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())