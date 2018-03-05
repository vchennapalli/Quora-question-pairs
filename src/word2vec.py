from keras.models import Model
from keras.layers import Input, Dense, Reshape, merge
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence

import urllib
import collections
import os
import re

import numpy as np
import tensorflow as tf
import pandas as pd

#FILEPATHS
TRAIN_CSV = "~/raw_data/train.csv"
TEST_CSV = "~/raw_data/test.csv"

#LOADS TRAINING AND TEST SET
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

"""preprocesses and converts question to a list of words"""
def question_to_wordlist(text):
    """
    input: string of text
    output: list of words
    """
    text = str(text)
    text = text.lower()

    #text cleaning and substitutions
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.split()

    return text

"""creates the collection of all the words in the datasets and returns """
def read_data(files):
    """
    input: files - list
    output: all the word from the train and test datafiles - list
    """
    vocabulary = []
    for dataset in files:
        for index, row in dataset.iterrows():
            for question in question_cols:
                vocabulary.extend(question_to_wordlist(row[question]))
    return vocabulary

"""processes raw input vocabulary into structured dataset"""
def build_dataset(words, n_words):
    """
    input: words - list, dictionary size - int
    output: data - list, freq - list of 2 unit components, dictionary - dict, inverse_dictionary - dict
    """
    freq = [['UKN', -1]]
    freq.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, count in freq:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    freq[0][1] = unk_count
    inverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, freq, dictionary, inverse_dictionary

"""performs primary preprocessing of data and returns polished data"""
def collect_data(vocabulary_size=100000):
    """
    input: vocab_size - int
    output: data - list, freq - list of 2 unit components, dictionary - dict, inverse_dictionary - dict
    """
    files = [train_df]#, test_df]
    vocabulary = read_data(files)
    print(vocabulary[:10])
    data, freq, dictionary, inverse_dictionary = build_dataset(vocabulary, vocabulary_size)
    del vocabulary
    return data, freq, dictionary, inverse_dictionary

vocab_size = 100 #100000
question_cols = ['question1', 'question2']
data, freq, dictionary, inverse_dictionary = collect_data(vocabulary_size=vocab_size)
print(data[:10])

window_size = 3
vector_dim = 30 #300
epochs = 5000 #1000000

valid_size = 5 #16
valid_window = 10 #100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

sampling_table = sequence.make_sampling_table(vocab_size)
couples, labels = skipgrams(data, vocab_size, window_size=window_size, sampling_table=sampling_table)
word_target, word_context = zip(*couples)
word_target = np.array(word_target, dtype="int32")
word_context = np.array(word_context, dtype="int32")

print(couples[:10], labels[:10])

input_target, input_context = Input((1,)), Input((1,))
embedding = Embedding(vocab_size, vector_dim, input_length=1, name='embedding')
target = embedding(input_target)
target = Reshape((vector_dim, 1))(target)
context = embedding(input_context)
context = Reshape((vector_dim, 1))(context)

similarity = merge([target, context], mode='cos', dot_axes=0)

dot_product = merge([target, context], mode='dot', dot_axes=1)
dot_product = Reshape((1,))(dot_product)

output = Dense(1, activation='sigmoid')(dot_product)

model = Model(input=[input_target, input_context], output=output)
model.compile(loss='binary_crossentropy', optimizer='rmsprop')

validation_model = Model(input=[input_target, input_context], output=similarity)

class SimilarityCallback:
    def run_sim(self):
        for i in range(valid_size):
            valid_word = inverse_dictionary[valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            sim = self._get_sim(valid_examples[i])
            nearest = (-sim).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = inverse_dictionary[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)

    @staticmethod
    def _get_sim(valid_word_idx):
        sim = np.zeros((vocab_size,))
        in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
        in_arr1[0,] = valid_word_idx
        for i in range(vocab_size):
            in_arr2[0,] = i
            out = validation_model.predict_on_batch([in_arr1, in_arr2])
            sim[i] = out
        return sim

sim_cb = SimilarityCallback()

arr_1 = np.zeros((1,))
arr_2 = np.zeros((1,))
arr_3 = np.zeros((1,))
for cnt in range(epochs):
    idx = np.random.randint(0, len(labels)-1)
    arr_1[0,] = word_target[idx]
    arr_2[0,] = word_context[idx]
    arr_3[0,] = labels[idx]
    loss = model.train_on_batch([arr_1, arr_2], arr_3)
    if cnt % 10 == 0: #100
        print("Iteration {}, loss={}".format(cnt, loss))
    if cnt % 100 == 0: #10000
        sim_cb.run_sim()

embedding_matrix = [[0] * vector_dim] + embedding.get_weights()[0]
print(embedding_matrix)
np.savetxt('embedding_matrix.txt', embedding_matrix)
print(np.loadtxt('embedding_matrix.txt'))
