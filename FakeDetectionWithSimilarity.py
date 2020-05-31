import math

import numpy as np
import scipy.spatial
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import nltk
import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import *

import json

import string
import Score

from sklearn.metrics import confusion_matrix
nltk.download('stopwords')


def load_glove_model(file):
    print("Loading Glove Model")
    f = open(file, 'r', encoding="utf8")
    glove_model = {}
    for line in f:
        split_lines = line.split()
        word = split_lines[0]
        word_embedding = np.array([float(value) for value in split_lines[1:]])
        glove_model[word] = word_embedding
    print(len(glove_model), " words loaded!")
    return glove_model


def remove_punctuation(raw_str):
    raw_str = raw_str.replace("\n", " ")
    raw_str = raw_str.replace("\\n", " ")
    raw_str = raw_str.replace(".", " ")
    raw_str = raw_str.replace(")", " ")
    raw_str = raw_str.replace("(", " ")
    raw_str = raw_str.replace("*", " ")
    raw_str = raw_str.replace("?", " ")
    raw_str = raw_str.replace("!", " ")
    raw_str = raw_str.replace("[", " ")
    raw_str = raw_str.replace("]", " ")
    raw_str = raw_str.replace("=", " ")
    raw_str = raw_str.replace("-", " ")
    raw_str = raw_str.replace("&", " ")
    raw_str = raw_str.replace(",", " ")
    raw_str = raw_str.replace(":", " ")
    raw_str = raw_str.replace(";", " ")
    raw_str = raw_str.replace("\"", " ")
    raw_str = raw_str.replace("#", " ")
    raw_str = raw_str.replace("\'", " ")
    raw_str = raw_str.replace("`", " ")
    raw_str = raw_str.replace("’", " ")
    raw_str = raw_str.replace("“", " ")
    raw_str = raw_str.replace("”", " ")

    return raw_str


def cosine_distance_wordembedding_method(embd_model, w1, w2):
    # vector_1 = np.mean([model[word] for word in preprocess(s1)],axis=0)
    # vector_2 = np.mean([model[word] for word in preprocess(s2)],axis=0)
    v1 = []
    v2 = []
    for word in w1:
        try:
            v1.append(embd_model[word])
        except:
            1 + 1
    for word in w2:
        try:
            v2.append(embd_model[word])
        except:
            1 + 1
    vector_1 = np.mean(v1, axis=0)
    vector_2 = np.mean(v2, axis=0)
    cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
    # print('Word Embedding method with a cosine distance asses that our two sentences are similar to',round((1-cosine)*100,2),'%')
    return round((1-cosine)*100, 2)


model = load_glove_model('data/glove.6B.100d.txt')

# JSON file
f = open('data/data.json', "r", encoding="utf8")

# Reading from file
data = json.loads(f.read())

# Iterating through the json
# list
fact = ""
tag = ""
author = ""
text_data = ""


def normalize_text(text):
    stop_words = set(stopwords.words('english'))
    stammer = SnowballStemmer("english")
    # normalize fact
    text = remove_punctuation(text)
    text = ' '.join(word for word in text.split() if word not in stop_words)  # remove stopwords from text
    text = stammer.stem(text)  # stemm
    return text


d = dict()

counter = 0
test_data = []
for i in data:
    json_data = i
    fact = normalize_text(json_data["fact"])
    tag = json_data["tag"]
    author = json_data["author"]
    text_data = normalize_text(json_data["data"])

    value = cosine_distance_wordembedding_method(model, fact.split(), text_data.split())
    try:
        d[tag].append(value)
    except:
        d[tag] = [value]

    if counter < (len(data) * 80) / 100:
        try:
            d[json_data["tag"]].append(value)
        except:
            d[json_data["tag"]] = [value]
    else:
        test_data.append({
            "tag": json_data["tag"],
            "similarity": value
        })
    counter += 1

avg = {}
for k in d.keys():
    sum = 0
    for n in d[k]:
        if math.isnan(n) == False:
            sum += n
        # else:
        #     print(n)
    avg[k] = sum / len(d[k])
print(avg)

all_count = 0
true_count = 0

y_true = []
y_pred = []

for td in test_data:
    min_sim = 999999
    min_tag = None
    for k in avg.keys():
        if min_sim > math.fabs(avg[k] - td["similarity"]):
            min_tag = k
            min_sim = math.fabs(avg[k] - td["similarity"])

    if min_tag == td["tag"]:
        true_count = true_count + 1
    y_true.append(td["tag"])
    y_pred.append(min_tag)
    all_count = all_count + 1

labels = ['fire', 'false', 'half-true', 'mostly-false', 'mostly-true', 'true', 'full-flop', 'half-flip', 'no-flip']
cm = confusion_matrix(y_true, y_pred, labels=labels)
print(cm)
print(cm.ravel())
Score.calculate_scores(labels, cm)

# Closing file
f.close()
