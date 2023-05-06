# 1.bert

#pip install -U tensorflow-text==2.6.0

import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tensorflow_text as text
from sklearn.metrics import accuracy_score  

d = pd.read_csv('/content/drive/MyDrive/aidos.csv')  
dsample = d1.sample(d2.shape[0]) #Return a random sample of items from an axis of object.
dconcat = pd.concat([dsample,d2])


X_train, X_test, Y_train, Y_test = train_test_split(dconcat['text'], dconcat['res'], stratify=dconcat['res'])

preprocess = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')
encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/4')
input = tf.keras.layers.Input(shape=(), name='text', dtype=tf.string)
text = preprocess(input)
output = encoder(text)
layer = tf.keras.layers.Dropout(0.1,name='dropout')(output['pooled_output'])
layer = tf.keras.layers.Dense(1, activation='sigmoid',name='output')(layer)
model = tf.keras.Model(inputs=[input],outputs=[layer])
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics='accuracy')
model.fit(X_train,Y_train,epochs=10,batch_size=5)
model.evaluate(X_test,Y_test)
Y_pred = model.predict(X_test)
Y_pred = [0 if val < 0.5 else 1 for val in Y_pred]
accuracy_score(Y_test, Y_pred)
print(model.predict(['zones']))
model.save('Bmodel')
from keras.models import load_model
model1 = load_model('/content/Bmodel')
print(model1.predict(['memswap']))