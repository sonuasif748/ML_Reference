from _csv import writer
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wordninja
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn import metrics
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
from fuzzywuzzy import fuzz
from sklearn.linear_model import SGDClassifier
from spellchecker import SpellChecker
spell = SpellChecker()

app = FastAPI()

class Testdata(BaseModel):
    field_name:str
class Traindata(BaseModel):
    key_name:str
    value_name:str


@app.post("/FakerType")
def faker_predict(reqdata:List):
    model = joblib.load('fakermodel2')
    res = dict()
    count_vect = model.named_steps.vect
    avoid_wordninja = ['fname','lname','mname','f name','l name','m name']
    # datap1 = pd.read_csv('newFakerData2_WN.csv')
    # list_r = [i for i in reqdata for j in datap1['text_wn'] if fuzz.ratio(i.lower(), j) > 50]
    for i in reqdata:
        # corrected_word = spell.correction(i)
        item = re.sub('[^A-Za-z0-9]+', ' ', i).lower()
        if item in avoid_wordninja:
            r = count_vect.transform([item])[0]
        else:
            t = wordninja.split(item)
            r = count_vect.transform([" ".join((t))])[0]
        val = model.named_steps.clf.predict(r)[0]
        res[i]=val
    return res


@app.post("/ReTrain")
def faker_retraining_process(traind:Traindata):
    list_data = [traind.key_name, traind.value_name]
    # with open('Fakers_OverSample_Data(sorted).csv', 'a', newline='') as f_object:
    #     writer_object = writer(f_object)
    #     writer_object.writerow(list_data)
    #     f_object.close()
    df = pd.read_csv('newFakerData(latest_Over).csv')
    X_train, X_test, y_train, y_test = train_test_split(df['text_wn'].values, df['label'].values, random_state=42)
    model = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf',
                     SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)),
                    ])
    model.fit(X_train, y_train)
    ypred = model.predict(X_test)
    print(metrics.accuracy_score(y_test,ypred))
    joblib.dump(model, 'fakermodel2')
    return {'Model Training Status': 'Successful'}

