from fastapi import FastAPI
import pandas as pd
import numpy as np
import joblib
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline  # creates a machine learning pipeline
from sklearn.naive_bayes import MultinomialNB  # to apply Naive Bayes Classifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # accuracy metrics
from sklearn.feature_extraction.text import CountVectorizer  # for BoW
from typing import List
from imblearn.under_sampling import RandomUnderSampler

data = pd.read_csv('test.csv')

# Undersampling
x = data.names.values.reshape(-1, 1)
y = data.boolean.values
ros = RandomUnderSampler(random_state=42)
x_ros, y_ros = ros.fit_resample(x, y)
df = pd.DataFrame({'names': x_ros.flatten(), 'boolean': y_ros.flatten()})
X_train, X_test, y_train, y_test = train_test_split(df['names'], df['boolean'], random_state=42)
df.to_csv('test.csv')

# OverSampling
rus = RandomOverSampler(random_state=0)
X_rus, Y_rus = rus.fit_resample(df, df['names'])
X_rus.to_csv('test2.csv', index=0)

# Multinomial and CV in pipeline
X_train, X_test, y_train, y_test = train_test_split(data['names'], data['boolean'], random_state=42)
cv = CountVectorizer()
model = Pipeline([('vect', cv), ('clf', MultinomialNB())])
model.fit(X_train, y_train)
ypred = model.predict(X_test)
joblib.dump(model, 'testmodel')  # create a model file

# FastAPI predict 
@app.post("/test")
async def testapi(adosinfo: List):
    d = dict()
    print('MultinomialNB', accuracy_score(y_test, ypred))
    for item in adosinfo:
        result = model.predict([item])
        if result == 1:
            d[item] = True
        else:
            d[item] = False
    return d