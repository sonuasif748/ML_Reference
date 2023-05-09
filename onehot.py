import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from keras.layers import Embedding
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential




nltk.download('punkt')

df = pd.read_csv('aidos_newtext.csv')

tokenizer = Tokenizer(num_words=1000, oov_token='OOV')
tokenizer.fit_on_texts(df['new_text2'])

df['tokenize_data'] = df.apply(lambda row: word_tokenize(row['new_text2']), axis=1)
df['sequences'] = tokenizer.texts_to_sequences(df['tokenize_data'])

df['sequences2'] = [','.join(map(str, l)) for l in df['sequences']]

x = df['sequences']

sent_length=8
embedded_docs=keras.preprocessing.sequence.pad_sequences(x,padding='pre',maxlen=sent_length)

y_ = (df.res).to_numpy().reshape(-1,1)
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y_)
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20)
input = ['count']

a=1