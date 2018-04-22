from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools
import datetime

from time import time
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, LSTM, Merge, CuDNNLSTM
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Input, Dense, Reshape, merge, concatenate
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import skipgrams
from nltk.corpus import stopwords
from keras.preprocessing import sequence
from keras.models import model_from_json

import urllib
import os

import numpy as np
import tensorflow as tf
import keras.backend as K

from math import *
from decimal import Decimal
from scipy.spatial import distance


#train_df and test_df are processed and imported from sentenceToWordList
from sentenceToWordList import *

#loading the inverse dictionary
inverse_dictionary = np.load(COMPUTE_DATA_PATH + 'g_inverse_dictionary.npy').item()
for key, value in inverse_dictionary.items():
	inverse_dictionary[key] = value.encode('ascii')

embeddingsMatrix = np.loadtxt(COMPUTE_DATA_PATH + 'g_embedding_matrix.txt')

#populating the dictionary
dictionary = {}

for index in range(len(inverse_dictionary)):
	dictionary[inverse_dictionary[index].decode("utf-8")] = index

print(train_df.shape)
print(test_df.shape)

#converting the questions into number vectors
for dataTuple in [train_df, test_df]:
	for index, row in dataTuple.iterrows():
		for question in question_cols:
			numVector = []
			for word in question_to_wordlist(row[question]):
				if (word in dictionary):
					numVector.append(dictionary[word])
			if (len(numVector) > maxSeqLength):
				numVector = numVector[0:maxSeqLength]
			dataTuple.set_value(index, question, numVector)

print("vector generation done!")

validation_size = 4
reqColumns= ['question1', 'question2', 'min_freq', 'common_neighbours', 'q_len1', 'q_len2', 'diff_len', 'word_len1', 'word_len2', 'common_words']
xTrain, xValidation, yTrain, yValidation = train_test_split(train_df[reqColumns],
								train_df['is_duplicate'], test_size=validation_size)

xTrain = [xTrain.question1, xTrain.question2, xTrain.min_freq, xTrain.common_neighbours, xTrain.q_len1, xTrain.q_len2, xTrain.diff_len,
			xTrain.word_len1, xTrain.word_len2, xTrain.common_words]
xValidation = [xValidation.question1, xValidation.question2,  xValidation.min_freq, xValidation.common_neighbours, xValidation.q_len1,
			xValidation.q_len2, xValidation.diff_len, xValidation.word_len1, xValidation.word_len2, xValidation.common_words]
for dataTuple in [xTrain, xValidation]:
	for i in range(2):
		dataTuple[i] = pad_sequences(dataTuple[i], maxlen=maxSeqLength)

print("sequences padded")
# Model variables
n_hidden = 50
gradientClippingNorm = 1.25
batch_size = 64
#keep n_epoch a multiple of 5
n_epoch = 1
embedding_dim = 300

#definations to calculate distance between two vectors
def euclidean_distance(l,r):
 	return K.sqrt(K.sum(K.square(l - r), axis=-1, keepdims=True))

def manhattan_distance(l, r):
	abs_v = K.abs(l-r)
	sum_v = K.sum(abs_v,axis=1,keepdims=True)
	return K.exp(-sum_v)

#describing the shape of various inputs

leftInput = Input(shape=(maxSeqLength,), dtype='int32')
rightInput = Input(shape=(maxSeqLength,), dtype='int32')
minFreq = Input(shape=(1,), dtype='float32')
commonNeigh = Input(shape=(1,), dtype='float32')
q_len1 = Input(shape=(1,), dtype='float32')
q_len2 = Input(shape=(1,), dtype='float32')
diff_len = Input(shape=(1,), dtype='float32')
word_len1 = Input(shape=(1,), dtype='float32')
word_len2 = Input(shape=(1,), dtype='float32')
common_words = Input(shape=(1,), dtype='float32')

#embedding layer to generate word embeddings
embeddingLayer = Embedding(len(embeddingsMatrix), embedding_dim, weights=[embeddingsMatrix], input_length=maxSeqLength, trainable=False)

encodedLeft = embeddingLayer(leftInput)
encodedRight = embeddingLayer(rightInput)

#LSTM layer
sharedLstm = CuDNNLSTM(n_hidden)

#to grab outputs from LSTM
leftOutput = sharedLstm(encodedLeft)
rightOutput = sharedLstm(encodedRight)

#using manhattan_distance as similarity Function
#it takes the two output vectors from the LSTMs and produces the distance as output
dist = Merge(mode=lambda x: manhattan_distance(x[0], x[1]), output_shape=lambda x: (x[0][0], 1))([leftOutput, rightOutput])

output = concatenate([dist, minFreq, commonNeigh, q_len1, q_len2, diff_len, word_len1, word_len2, common_words])
output = Dense(1, activation = 'sigmoid')(output)

#model for the siamese LSTM
siameseLSTM = Model([leftInput, rightInput, minFreq, commonNeigh, q_len1, q_len2, diff_len, word_len1, word_len2, common_words], [output])
optimizer = Adadelta(clipnorm=gradientClippingNorm)

siameseLSTM.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

#training begings
training_start_time = time()

print("Started training")
for i in range(n_epoch):
	siameseLSTMTrained = siameseLSTM.fit([xTrain[0], xTrain[1], xTrain[2], xTrain[3], xTrain[4], xTrain[5], xTrain[6], xTrain[7],
						xTrain[8], xTrain[9]], yTrain.values, batch_size=batch_size, epochs=4,
                            	validation_data=([xValidation[0], xValidation[1], xValidation[2], xValidation[3], xValidation[4], xValidation[5],
					xValidation[6], xValidation[7], xValidation[8], xValidation[9]], yValidation.values))

	siameseLSTM_JSON = siameseLSTM.to_json()
	with open("../models/siameseLSTM_JSON.json","w") as json_file:
		json_file.write(siameseLSTM_JSON)
	siameseLSTM.save_weights("../models/siameseLSTM_WEIGHTS.h5")
	siameseLSTM.save("../models/siameseLSTM_model.h5")
	#simaeseLSTM = load_model(COMPUTE_DATA_PATH + 'siameseLSTM.h5')
	print("Finished epochs "+ repr(i+1))

print("Training time finished.\n{} epochs in {}".format(n_epoch, datetime.timedelta(seconds=time()-training_start_time)))

xTrain = [np.array(test_df['question1'].tolist()), np.array(test_df['question2'].tolist()), np.array(test_df['min_freq'].tolist()),
		np.array(test_df['common_neighbours'].tolist()), np.array(test_df['q_len1'].tolist()), np.array(test_df['q_len2'].tolist()),
		np.array(test_df['diff_len'].tolist()), np.array(test_df['word_len1'].tolist()), np.array(test_df['word_len2'].tolist()),
		np.array(test_df['common_words'].tolist())]

for dataTuple in [xTrain]:
	for i in range(2):
		dataTuple[i] = pad_sequences(dataTuple[i], maxlen=maxSeqLength)

# load json and create model
#json_file = open(MODELS_PATH + 'siameseLSTM_JSON.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
# load weights into new model
#loaded_model.load_weights(MODELS_PATH + "siameseLSTM_WEIGHTS.h5")

#print("Loaded model from disk")

#getting the predictions from the trained_model
predictions = siameseLSTM.predict([xTrain[0],xTrain[1], xTrain[2], xTrain[3], xTrain[4], xTrain[5], xTrain[6], xTrain[7], xTrain[8], xTrain[9]])
print("predictions ready")
print("Geerating sub file")
import pandas as pdn

#writing the predictions to a csv file for kaggle submission
sub_df = pd.DataFrame(data=predictions,columns={"is_duplicate"})
sub_df.to_csv(path_or_buf="../results/kaggle_sub.csv", columns={"is_duplicate"}, header=True, index=True, index_label="test_id")
