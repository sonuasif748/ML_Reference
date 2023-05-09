import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import nltk
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
import tensorflow as tf
from tensorflow import keras
import joblib
import keras
from sklearn import pipeline
from sklearn import preprocessing
from keras.wrappers.scikit_learn import KerasClassifier

nltk.download('punkt')

df = pd.read_csv("Entry_cleaned.csv")
df.replace({"response":{200:1,401:0}},inplace=True)
tokenizer = Tokenizer(num_words=1000, oov_token='OOV')
df['path'] = df['path'].astype(str).str.lower()
tokenizer.fit_on_texts(df['path'])
df['tokenize_data'] = df.apply(lambda row: word_tokenize(row['path']), axis=1)
df['tokenize_data'] = df['tokenize_data'].astype(str).str.lower()

X_train, X_test,Y_train,Y_test = train_test_split(df['tokenize_data'],df['response'],test_size=0.2,random_state=5)

model = keras.Sequential([
    keras.layers.Dense(26, input_shape=(26,), activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
Xcnn_train = tokenizer.texts_to_sequences(X_train)
Xcnn_test = tokenizer.texts_to_sequences(X_test)
vocab_size = len(tokenizer.word_index) + 1

from keras_preprocessing.sequence import pad_sequences
maxlen = 100
Xcnn_train = pad_sequences(Xcnn_train, padding='post', maxlen=maxlen)
Xcnn_test = pad_sequences(Xcnn_test, padding='post', maxlen=maxlen)
print(Xcnn_train[0, :])

from keras.models import Sequential
from keras import layers
embedding_dim = 200
textcnnmodel = Sequential()
textcnnmodel.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
textcnnmodel.add(layers.Conv1D(128, 5, activation='relu'))
textcnnmodel.add(layers.GlobalMaxPooling1D())
textcnnmodel.add(layers.Dense(10, activation='relu'))
textcnnmodel.add(layers.Dense(1, activation='sigmoid'))
textcnnmodel.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
textcnnmodel.summary()

textcnnmodel.fit(Xcnn_train, Y_train,
                    epochs=10,
                    verbose=False,
                    validation_data=(Xcnn_test, Y_test),
                    batch_size=10)
loss, accuracy = textcnnmodel.evaluate(Xcnn_train, Y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = textcnnmodel.evaluate(Xcnn_test, Y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

test1 = tokenizer.texts_to_sequences(['size','name','asif'])
test1 = pad_sequences(test1, padding='post', maxlen=maxlen)
a = textcnnmodel.predict(test1)

def model():
    model = Sequential([
        layers.Dense(64, input_dim=Xcnn_train.shape[1], activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')

pipe = pipeline.Pipeline([
    ('rescale', preprocessing.StandardScaler()),
    ('nn', KerasClassifier(build_fn=model, nb_epoch=10, batch_size=128,
                           validation_split=0.2, callbacks=[early_stopping]))
])

pipe.fit(Xcnn_train, Y_train.values)

pipe.predict_proba(Xcnn_test)

pipe.predict(Xcnn_test)

test1 = tokenizer.texts_to_sequences(['size','name','asfyufguivkvhjhif'])
test1 = pad_sequences(test1, padding='post', maxlen=maxlen)
pipe.predict(test1)