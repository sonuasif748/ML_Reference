#pip install autokeras
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_files
import autokeras as ak
from sklearn import metrics


df = pd.read_csv('/content/CompanyTag(new)_OS.csv')
df=df.dropna()
df = df.sample(frac=1)

x = df['Title']
y = df[['Tag']]
y.nunique()
df[df['Tag']=='architecture'].count()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=30)
x_train = x_train.to_numpy()
x_test = x_test.to_numpy()
y_train =y_train.to_numpy()
y_test= y_test.to_numpy()
# sdv

# Initialize the text classifier.
clf = ak.TextClassifier(
    overwrite=True, max_trials=1)  # It only tries 1 model as a quick demo.# Feed the text classifier with training data.
clf.fit(x_train, y_train, epochs=1)# Predict with the best model.
predicted_y = clf.predict(x_test)# Evaluate the best model with testing data.
print(clf.evaluate(x_test, y_test))

print("Accuracy:", metrics.accuracy_score(y_test, predicted_y))
print(classification_report(y_test,predicted_y))
print("Accuracy:", metrics.accuracy_score(y_test, predicted_y))
l = np.array(['Convenience Store amp Gas Station Circle K', 'Conflict Resolution Center for Conflict Resolution Chicago','JPC Architects','Xitrix Computer Corporation']).reshape(-1,1)
clf.predict(l)