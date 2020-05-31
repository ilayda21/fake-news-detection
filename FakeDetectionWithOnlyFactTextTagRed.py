# JSON file
import json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf
from keras import backend as K
K.clear_session()


f = open('data/data.json', "r", encoding="utf8")


def get_tag(tag):
    tg = None
    if tag == "fire" or tag == "false" or tag == "mostly-false":
        tg = "false"
    elif tag == "mostly-true" or tag == "true":
        tg = "true"
    elif tag == "half-true":
        tg = "half-true"
    return tg


# Reading from file
data = json.loads(f.read())
facts = []
tags = []
tag_id = 0
tag_dict = {}
for i in data:
    json_data = i
    t = json_data["tag"]
    t = get_tag(t)
    if t is not None:
        facts.append(json_data["fact"])

        if t not in tag_dict:
            tag_dict[t] = tag_id
            tag_id = tag_id + 1

        tags.append(tag_dict[t])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(facts)
word_index = tokenizer.word_index
vocab_size = len(word_index)
print("vocab size: " + str(vocab_size))

sequences = tokenizer.texts_to_sequences(facts)
padded = pad_sequences(sequences, maxlen=500, padding='post', truncating='post')

print(padded.shape)

split = 0.2
split_n = int(round(len(padded)*(1-split),0))

train_data = np.array(padded[:split_n])
train_labels = np.array(tags[:split_n])
test_data = np.array(padded[split_n:])
test_labels = np.array(tags[split_n:])

print("train labels: " + str(len(train_labels)))

print("train data: " + str(len(train_data)))

print("test labels: " + str(len(test_labels)))

print("test data: " + str(len(test_data)))

embeddings_index = {}
with open('data/glove.6B.100d.txt', encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print("glove is read  - " + str(len(coefs)))

embeddings_matrix = np.zeros((vocab_size+1, 100))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector

print("Embedding matrix created.")

# Build the architecture of the model

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size + 1, 100, weights=[embeddings_matrix], trainable=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.LSTM(20, return_sequences=True),
    tf.keras.layers.LSTM(20),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

history = model.fit(train_data, train_labels, epochs=20, batch_size=100, validation_data=[test_data, test_labels])

print("Training Complete")

# Visualize the results:

print(history.history['accuracy'])