import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

df1 = pd.read_csv("split2.csv")
a=1


tokenizer1 = Tokenizer(num_words=None, char_level=True, oov_token='<OOV>')
tokenizer1.fit_on_texts(df1['Title'])

sequences1 = tokenizer1.texts_to_sequences(df1['Title'])
padded1 = pad_sequences(sequences1)

industry_labels1 = df1['Tag'].unique()
industry_labels_dict1 = { industry_labels1[i]: i for i in range(len(industry_labels1))}
df1['Tag'] = df1['Tag'].map(industry_labels_dict1)

X_train1, X_test1, y_train1, y_test1 = train_test_split(padded1, df1['Tag'], test_size=0.25, random_state=0)

model1 = Sequential()
model1.add(Embedding(input_dim=len(tokenizer1.word_index) + 1, output_dim=32, input_length=padded1.shape[1]))
model1.add(LSTM(units=100, dropout=0.2, recurrent_dropout=0.2))
model1.add(Dense(units=len(industry_labels1), activation='softmax'))
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

y_train_categorical1 = to_categorical(y_train1)
y_test_categorical1 = to_categorical(y_test1)

model1.fit(X_train1, y_train_categorical1, epochs=1, batch_size=32)

loss, accuracy = model1.evaluate(X_test1, y_test_categorical1, verbose=0)
print("Accuracy: %.2f%%" % (accuracy*100))

model1.save('model2.h5')



# fastapi

app=FastAPI()

@app.post("/companypredict")
def company_predict(reqdata:List):
    m1 = keras.models.load_model('model1.h5')
    m2 = keras.models.load_model('model2.h5')
    m3 = keras.models.load_model('model3.h5')
    m4 = keras.models.load_model('model4.h5')
    df = pd.read_csv('final.csv')
    res = dict()
    tokenizer = Tokenizer(num_words=None, char_level=True, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['Title'])
    industry_labels = df['Tag'].unique()
    industry_labels_dict = {industry_labels[i]: i for i in range(len(industry_labels))}

    res = {'Model-1':[], 'Model-2':[], 'Model-3':[], 'Model-4':[], 'stack':[]}

    for i in reqdata:
        sequence = tokenizer.texts_to_sequences([i])

        padded_input1 = pad_sequences(sequence, maxlen=m1.input.shape[1])
        prediction1 = m1.predict(padded_input1)
        predicted_industry_index1 = np.argmax(prediction1)
        predicted_industry1 = [key for key, value in industry_labels_dict.items() if value == predicted_industry_index1][0]
        res['Model-1'].append({i:predicted_industry1})

        padded_input2 = pad_sequences(sequence, maxlen=m2.input.shape[1])
        prediction2 = m2.predict(padded_input2)
        predicted_industry_index2 = np.argmax(prediction2)
        predicted_industry2 = [key for key, value in industry_labels_dict.items() if value == predicted_industry_index2][0]
        res['Model-2'].append({i:predicted_industry2})

        padded_input3 = pad_sequences(sequence, maxlen=m3.input.shape[1])
        prediction3 = m3.predict(padded_input3)
        predicted_industry_index3 = np.argmax(prediction3)
        predicted_industry3 = [key for key, value in industry_labels_dict.items() if value == predicted_industry_index3][0]
        res['Model-3'].append({i:predicted_industry3})

        padded_input4 = pad_sequences(sequence, maxlen=m4.input.shape[1])
        prediction4 = m4.predict(padded_input4)
        predicted_industry_index4 = np.argmax(prediction4)
        predicted_industry4 = [key for key, value in industry_labels_dict.items() if value == predicted_industry_index4][0]
        res['Model-4'].append({i:predicted_industry4})

        combined = hstack((prediction1, prediction2, prediction3, prediction4))
        predicted_industry = [key for key, value in industry_labels_dict.items() if value == round(np.argmax(combined)/4)][0]
        res['stack'].append({i:predicted_industry})

    return res