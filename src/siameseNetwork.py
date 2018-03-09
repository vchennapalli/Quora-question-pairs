from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools
import datetime

from time import time
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, LSTM, Merge
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Input, Dense, Reshape, merge, concatenate
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import skipgrams
from nltk.corpus import stopwords
from keras.preprocessing import sequence

import urllib
import os

import numpy as np
import tensorflow as tf
from sentenceToWordList import *

inverse_dictionary = np.load(COMPUTE_DATA_PATH + 'inverse_dictionary.npy').item()
for key, value in inverse_dictionary.iteritems():
	inverse_dictionary[key] = value.encode('ascii')

embeddingsMatrix = np.loadtxt(COMPUTE_DATA_PATH + 'embedding_matrix.txt')

dictionary = {}
maxSeqLength = 0

for index in range(len(inverse_dictionary)):
	dictionary[inverse_dictionary[index]] = index
	
for dataTuple in [train_df, test_df]:
	for index, row in dataTuple.iterrows():
		for question in question_cols:
			numVector = []
			for word in question_to_wordlist(row[question]):
				if (word in dictionary):
					numVector.append(dictionary[word])
			dataTuple.set_value(index, question, numVector)
			maxSeqLength = max(maxSeqLength, len(numVector))

del test_df

validation_size = 40000
xTrain, xValidation, yTrain, yValidation = train_test_split(train_df[question_cols], train_df['is_duplicate'], test_size=validation_size)

xTrain = [xTrain.question1, xTrain.question2]
xValidation = [xValidation.question1, xValidation.question2]
for dataTuple in [xTrain, xValidation]:
	for i in range(2):
		dataTuple[i] = pad_sequences(dataTuple[i], maxlen=maxSeqLength)

# Model variables
n_hidden = 50
gradientClippingNorm = 1.25
batch_size = 64
#keep n_epoch a multiple of 5
n_epoch = 50
embedding_dim = 300

leftInput = Input(shape=(maxSeqLength,), dtype='int32')
rightInput = Input(shape=(maxSeqLength,), dtype='int32')

embeddingLayer = Embedding(len(embeddingsMatrix), embedding_dim, weights=[embeddingsMatrix], input_length=maxSeqLength, trainable=False)

encodedLeft = embeddingLayer(leftInput)
encodedRight = embeddingLayer(rightInput)

sharedLstm = LSTM(n_hidden)

leftOutput = sharedLstm(encodedLeft)
rightOutput = sharedLstm(encodedRight)

output = concatenate([leftOutput, rightOutput])
output = Dense(1, activation = 'sigmoid')(output)

siameseLSTM = Model([leftInput, rightInput], [output])
optimizer = Adadelta(clipnorm=gradientClippingNorm)

siameseLSTM.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

training_start_time = time()

print("Started training")
for i in range(n_epoch/5):
	siameseLSTMTrained = siameseLSTM.fit([xTrain[0], xTrain[1]], yTrain.values, batch_size=batch_size, epochs=5,
                            	validation_data=([xValidation[0], xValidation[1]], yValidation.values))
	
	siameseLSTM.save(COMPUTE_DATA_PATH + 'siameseLSTM.h5')
	#simaeseLSTM = load_model(COMPUTE_DATA_PATH + 'siameseLSTM.h5')
	print("Finished epochs "+ repr((i+1)*5))

print("Training time finished.\n{} epochs in {}".format(n_epoch, datetime.timedelta(seconds=time()-training_start_time)))

plt.plot(siameseLSTMTrained.history['acc'])
plt.plot(siameseLSTMTrained.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(siameseLSTMTrained.history['loss'])
plt.plot(siameseLSTMTrained.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show() 
