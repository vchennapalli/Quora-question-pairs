from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense, Reshape, merge
from keras.preprocessing import sequence
from keras.preprocessing.sequence import skipgrams


import urllib
import collections
import os
import re

import numpy as np
import tensorflow as tf
from sentenceToWordList import *



"""creates the collection of all the words in the datasets and returns """
def readData(files):
    """
    input: files - list
    output: all the word from the train and test datafiles - list
    """
    vocab = []
    for d in files:
        for i, r in d.iterrows():
            for q in qCols:
                vocab.extend(question_to_wordlist(r[q]))
    return vocab

"""processes raw input vocabulary into structured dataset"""
def buildDataset(words):
    """
    input: words - list
    output: data - list, freq - list of 2 unit components, dictionary - dict, inverseDict - dict
    """
    freq = [['unknown', -1]]
    freq.extend(collections.Counter(words).most_common())
    dictionary = dict()
    for w, c in freq:
        dictionary[w] = len(dictionary)
    data = list()
    unk_count = 0
    for w in words:
        if w in dictionary:
            i = dictionary[w]
        else:
            i = 0
            unknown += 1
        data.append(i)
    freq[0][1] = unknown
    inverseDict = dict(zip(dictionary.values(), dictionary.keys()))
    return data, freq, dictionary, inverseDict

"""performs primary preprocessing of data and returns polished data"""
def collectDataset():
    """
    output: data - list, freq - list of 2 unit components, dictionary - dict, inverseDict - dict
    """
    files = [trainDatafile, Datafile]
    vocabulary = readData(files)
    print(vocabulary[:10])
    data, freq, dictionary, inverseDict = buildDataset(vocabulary)
    del vocabulary
    return data, freq, dictionary, inverseDict

data, freq, dictionary, inverseDict = collectDataset()

windowSize = 3
vectorDimen = 300 #30
epochs = 240000 #5000

valSize = 16 #5
valWindow = 100 #10
valExamples = np.random.choice(valWindow, valSize, replace=False)

sampleTable = sequence.make_sampling_table(vocabSize)
couples, labels = skipgrams(data, vocabSize, window_size=windowSize, sampling_table=sampleTable)
wordTarget, wordContext = zip(*couples)
wordTarget = np.array(wordTarget, dtype="int32")
wordContext = np.array(wordContext, dtype="int32")

inputTarget, inputContext = Input((1,)), Input((1,))
embedding = Embedding(vocabSize, vectorDimen, input_length=1, name='embedding')
target = embedding(inputTarget)
target = Reshape((vectorDimen, 1))(target)
context = embedding(inputContext)
context = Reshape((vectorDimen, 1))(context)

similarity = merge([target, context], mode='cos', dot_axes=0)

dot_product = merge([target, context], mode='dot', dot_axes=1)
dot_product = Reshape((1,))(dot_product)

output = Dense(1, activation='sigmoid')(dot_product)

model = Model(input=[inputTarget, inputContext], output=output)
model.compile(loss='binary_crossentropy', optimizer='rmsprop')

validationModel = Model(input=[inputTarget, inputContext], output=similarity)

class SimilarityCallback:
    def runSim(self):
        for i in range(valSize):
            valid_word = inverseDict[valExamples[i]]
            topK = 10 
            sim = self.getSim(valExamples[i])
            nearest = (-sim).argsort()[1:topK + 1]
            log_str = 'Nearest to %s:' % valid_word
            for i in range(topK):
                close_word = inverseDict[nearest[i]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)

    @staticmethod
    def getSim(validIdx):
        sim = np.zeros((vocabSize,))
        in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
        in_arr1[0,] = validIdx
        for i in range(vocabSize):
            in_arr2[0,] = i
            output = validationModel.predict_on_batch([in_arr1, in_arr2])
            sim[i] = output
        return sim

simCb = SimilarityCallback()

wordTargetArray = np.zeros((1,))
wordContextArray = np.zeros((1,))
labelsArray = np.zeros((1,))
for count in range(epochs):
    idx = np.random.randint(0, len(labels)-1)
    wordTargetArray[0,] = wordTarget[idx]
    wordContextArray[0,] = wordContext[idx]
    labelsArray[0,] = labels[idx]
    loss = model.train_on_batch([wordTargetArray, wordContextArray], labelsArray)
    if count % 100 == 0: #10
        print("Iteration {}, loss={}".format(count, loss))
    if count % 10000 == 0: #100
        simCb.runSim()
zerosRow = np.array([0] * vectorDimen)
zerosRow.shape = (1, vectorDimen)
embeddingMatrix = embedding.get_weights()[0]
#print(embeddingMatrix.shape)
embeddingMatrix = np.concatenate((zerosRow, embeddingMatrix), axis = 0)
#print(embeddingMatrix)
np.savetxt(COMPUTE_DATA_PATH + 'embedding_matrix.txt', embeddingMatrix, fmt="%.5f")
#np.savetxt('embeddingMatrix.txt', embeddingMatrix)

np.save(COMPUTE_DATA_PATH + 'inverse_dictionary.npy', inverseDict)

np.save(COMPUTE_DATA_PATH + 'dictionary.npy', dictionary)