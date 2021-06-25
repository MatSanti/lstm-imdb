import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import re
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import  confusion_matrix,classification_report

max_features = 5000 #numero maximo de palavras do dicionario
max_sequence_length = 300 #numero maximo do vetor tokenizado
embedding_dim = 128 #tamanho da dimensão de conversão da camada de embedding


df = pd.read_csv('IMDB_Dataset.csv')


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()  # lowercase text
    text = text.replace('<br />',' ')
    text = REPLACE_BY_SPACE_RE.sub(' ',text) 
    text = BAD_SYMBOLS_RE.sub('', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text

print(df['review'][0])
df['review'] = df['review'].apply(clean_text)

print(df['review'][0])



tokenizer = Tokenizer(num_words=max_features, split=' ')

tokenizer.fit_on_texts(df['review'].values)

X = tokenizer.texts_to_sequences(df['review'].values)
print(X[0])
X = pad_sequences(X, maxlen=max_sequence_length)

Y = pd.get_dummies(df['sentiment']).values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20)

model = Sequential()

model.add(Embedding(max_features, embedding_dim, input_length=X.shape[1]))

model.add(LSTM(max_sequence_length, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(Y.shape[1], activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

history = model.fit(X_train, Y_train, epochs=5, validation_split=0.1)
model.save_weights('my_model_weights_sentimentos.h5')


score = model.evaluate(X_test,Y_test)
print('Test score:', score[0])
print('Test accuracy:', score[1])

y_pred = model.predict(X_test)

y_test_class = np.argmax(Y_test, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)

print(classification_report(y_test_class, y_pred_class))
print(confusion_matrix(y_test_class, y_pred_class))

df_resultados = pd.DataFrame.from_dict(history.history)
df_resultados['accuracy'] = 100 * df_resultados['accuracy']


df_resultados['accuracy'].plot()
# df_resultados['loss'].plot()
plt.xlabel('Época')
plt.ylabel('% de Acertos')
plt.show()