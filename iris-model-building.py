import pandas as pd
iris = pd.read_csv('Iris.csv')

df = iris.copy()
target = 'Species'

target_mapper = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
def target_encode(val):
    return target_mapper[val]

df['Species'] = df['Species'].apply(target_encode)

X = df.drop(['Species', 'Id'], axis=1)
Y = df['Species']

# Random Forest Model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, Y)

# Saving model
import pickle
pickle.dump(clf, open('iris_clf.pkl', 'wb'))
