import math
import re

import numpy as np
import scipy.spatial
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import nltk
import csv
from ExpandContraction import expandContractions
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import *
import heapq
import json
from sklearn.metrics import confusion_matrix
import Score
import string

nltk.download('stopwords')

def remove_punctuation(raw_str):
    raw_str = raw_str.replace("’s", "")
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
    raw_str = raw_str.replace("’", "")
    raw_str = raw_str.replace("“", " ")
    raw_str = raw_str.replace("”", " ")

    return raw_str


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
    stop_words = set(stopwords)
    # normalize fact
    total = ""
    for w in text.split():
        s = expandContractions(w)
        if total == "":
            total = s
        else:
            total = total + " " + s

    total = re.sub(r'\[[0-9]*\]', ' ', remove_punctuation(total))
    total = re.sub(r'\s+', ' ', total)
    # Removing special characters and digits
    total = re.sub('[^a-zA-Z]', ' ', total)
    total = re.sub(r'\s+', ' ', total)
    total = remove_punctuation(total)

    total = ' '.join(wrd for wrd in total.split() if wrd not in stop_words)  # remove stopwords from text
    return total

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


def load_glove_model(file):
    print("Loading Glove Model")
    ff = open(file, 'r', encoding="utf8")
    glove_model = {}
    for line in ff:
        split_lines = line.split()
        wrd = split_lines[0]
        word_embedding = np.array([float(value) for value in split_lines[1:]])
        glove_model[wrd] = word_embedding
    print(len(glove_model), " words loaded!")
    return glove_model


def get_tag(tag):
    t = None
    if tag == "fire" or tag == "false" or tag == "mostly-false":
        t = "false"
    elif tag == "mostly-true" or tag == "true":
        t = "true"
    elif tag == "half-true":
        t = "half-true"
    return t


model = load_glove_model('data/glove.6B.100d.txt')

d = dict()
counter = 0
test_data = []
for i in data:
    json_data = i
    # json_data = json.loads(i)

    data_tag = get_tag(json_data["tag"])

    if data_tag is not None:
        # Removing Square Brackets and Extra Spaces
        article_text = re.sub(r'\[[0-9]*\]', ' ', json_data["data"])
        article_text = re.sub(r'\s+', ' ', article_text)
        # Removing special characters and digits
        formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text)
        formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

        sentence_list = nltk.sent_tokenize(article_text)

        stopwords = nltk.corpus.stopwords.words('english')
        word_frequencies = {}
        for word in nltk.word_tokenize(formatted_article_text):
            if word not in stopwords:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1
        if len(word_frequencies) is not 0:
            maximum_frequncy = max(word_frequencies.values())
        else:
            maximum_frequncy = 0

        for word in word_frequencies.keys():
            if maximum_frequncy != 0:
                word_frequencies[word] = (word_frequencies[word] / maximum_frequncy)

        sentence_scores = {}
        for sent in sentence_list:
            for word in nltk.word_tokenize(sent.lower()):
                if word in word_frequencies.keys():
                    if len(sent.split(' ')) < 40:
                        if sent not in sentence_scores.keys():
                            sentence_scores[sent] = word_frequencies[word]
                        else:
                            sentence_scores[sent] += word_frequencies[word]

        summary_sentences = heapq.nlargest(20, sentence_scores, key=sentence_scores.get)

        summary = ' '.join(summary_sentences)
        # print(summary)

        fact = normalize_text(json_data["fact"])
        normalized_summary = normalize_text(summary)

        value = cosine_distance_wordembedding_method(model, fact.split(), normalized_summary.split())
        if counter < (len(data) * 80) / 100:
            try:
                d[data_tag].append(value)
            except:
                d[data_tag] = [value]
        else:
            test_data.append({
                "tag": data_tag,
                "similarity": value
            })
        counter += 1

avg = {}
for k in d.keys():
    summ = 0
    for n in d[k]:
        if math.isnan(n) == False:
            summ += n
    avg[k] = summ / len(d[k])
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

# print(str(true_count) + " / " + str(all_count))
labels = ['true', 'half-true', 'false']
cm = confusion_matrix(y_true, y_pred, labels=labels)
print(cm)
print(cm.ravel())
Score.calculate_scores(labels, cm)
# Closing file
f.close()
