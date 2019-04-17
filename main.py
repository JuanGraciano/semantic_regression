#!/usr/bin/env python
# -*- coding: utf-8 -*-
import keras
import nltk
import pandas as pd
import numpy as np
import re

from nltk.tokenize import RegexpTokenizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

import gensim

import sys
reload(sys)
sys.setdefaultencoding('utf8')

def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, clean_questions, generate_missing=False, k=300):
    embeddings = clean_questions['tokens'].apply(lambda x: get_average_word2vec(x, vectors, generate_missing=generate_missing, k=k))
    return list(embeddings)

def get_metrics(y_test, y_predicted):  
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None,
                                    average='weighted')             
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None,
                              average='weighted')
    
    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')
    
    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1

def cv(data):
    count_vectorizer = CountVectorizer()

    emb = count_vectorizer.fit_transform(data)

    return emb, count_vectorizer

def sanitize_characters(raw, clean):    
    for line in raw:
        clean.write(line)

def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    return df

def getData(file_path, create=False):
    
    questions = pd.read_csv(file_path)
    questions.columns=['text', 'label']
    # print(questions.head())
    # print(questions.tail())
    # print(questions.describe())

    clean_questions = standardize_text(questions, "text")
    # output_file = open(new_file_path, "w")
    # sanitize_characters(questions, output_file)
    # questions.to_csv(new_file_path)
    # questions.head()

    tokenizer = RegexpTokenizer(r'\w+')
    
    clean_questions["tokens"] = clean_questions["text"].apply(tokenizer.tokenize)
    # print(clean_questions.head())

    all_words = [word for tokens in clean_questions["tokens"] for word in tokens]
    # print(len(all_words))
    sentence_lengths = [len(tokens) for tokens in clean_questions["tokens"]]
    # print(sentence_lengths)
    VOCAB = sorted(list(set(all_words)))
    print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))
    print("Max sentence length is %s" % max(sentence_lengths))

    # ====
    # Bag of Words Counts

    list_corpus = clean_questions["text"].tolist()
    list_labels = clean_questions["label"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.2, random_state=40)

    X_train_counts, count_vectorizer = cv(X_train)
    X_test_counts = count_vectorizer.transform(X_test)

    # ====
    # Enter word2vec
    word2vec_path = "GoogleNews-vectors-negative300.bin.gz"
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)


    # Dividir data entrenamiento
    embeddings = get_word2vec_embeddings(word2vec, clean_questions, k=word2vec.vector_size)
    X_train_word2vec, X_test_word2vec, y_train_word2vec, y_test_word2vec = train_test_split(embeddings, list_labels, test_size=0.2, random_state=40)

    # Entrenar modelo de regresion lineal
    clf_w2v = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', multi_class='multinomial', random_state=40)
    clf_w2v.fit(X_train_word2vec, y_train_word2vec)
    y_predicted_word2vec = clf_w2v.predict(X_test_word2vec)

    # Evaluar modelo: Accuracy, precision, recal y f1
    accuracy_word2vec, precision_word2vec, recall_word2vec, f1_word2vec = get_metrics(y_test_word2vec, y_predicted_word2vec)
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_word2vec, precision_word2vec, recall_word2vec, f1_word2vec))


    return VOCAB, clean_questions

if __name__ == "__main__":
    getData(".../NLC/data/nueva_data/transaccion/trans_100_clean.csv")
   