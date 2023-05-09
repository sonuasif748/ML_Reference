from fastapi import FastAPI
import joblib
import pandas as pd
from pip._vendor.urllib3.filepost import writer
from pydantic import BaseModel
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from csv import writer
from imblearn.over_sampling import RandomOverSampler
from typing import List
import wordninja
import re
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.tests.test_pprint import SimpleImputer, StandardScaler
from sklearn import metrics

# a = 1
app = FastAPI()

class Testdata(BaseModel):
    field_name:str
class Traindata(BaseModel):
    key_name:str
    value_name:str


@app.post("/FakerType")
def faker_type(reqdata:List):
    model = joblib.load('fakermodel')
    res = dict()
    count_vect = model.named_steps.vect
    for i in reqdata:
        item = re.sub('[^A-Za-z0-9]+', ' ', i).lower()
        t = wordninja.split(item)
        r = count_vect.transform([" ".join((t))])
        val = model.named_steps.clf.predict(r)[0]
        res[i]=val
    return res


@app.post("/PreTrain")
def retraining_process(traind:Traindata):
    list_data = [traind.key_name, traind.value_name]
    # with open('faker_dataset.csv', 'a', newline='') as f_object:
    #     writer_object = writer(f_object)
    #     writer_object.writerow(list_data)
    #     f_object.close()
    df = pd.read_csv('faker_dataset.csv', on_bad_lines='skip')
    # rus = RandomOverSampler(random_state=0)
    # X_rus, Y_rus = rus.fit_resample(df, df['Keys'])
    # X_rus.to_csv('fieldnames.csv', index=0)
    # df2 = pd.read_csv('fieldnames.csv')
    X_train, X_test, y_train, y_test = train_test_split(df['Keys'].values, df['Values'].values, test_size=0.20, random_state=0)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    # X_test = count_vect.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean', p=2)
    clf = knn.fit(X_train_counts, y_train)
    model = Pipeline([
        ('vect', CountVectorizer()),
        ('clf', knn)
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ac = metrics.accuracy_score(y_test, y_pred)
    print(ac)
    joblib.dump(model, 'fakermodel')
    return {'Model Training Status': 'Successful'}