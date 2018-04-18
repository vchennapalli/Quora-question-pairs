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
from sentenceToWordList import *

inverse_dictionary = np.load(COMPUTE_DATA_PATH + 'inverse_dictionary.npy').item()
for key, value in inverse_dictionary.items():
	inverse_dictionary[key] = value.encode('ascii')

embeddingsMatrix = np.loadtxt(COMPUTE_DATA_PATH + 'embedding_matrix.txt')

dictionary = {}

for index in range(len(inverse_dictionary)):
	dictionary[inverse_dictionary[index].decode("utf-8")] = index
	
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
n_epoch = 8
embedding_dim = 300

leftInput = Input(shape=(maxSeqLength,), dtype='int32')
rightInput = Input(shape=(maxSeqLength,), dtype='int32')

embeddingLayer = Embedding(len(embeddingsMatrix), embedding_dim, weights=[embeddingsMatrix], input_length=maxSeqLength, trainable=False)

encodedLeft = embeddingLayer(leftInput)
encodedRight = embeddingLayer(rightInput)

sharedLstm = CuDNNLSTM(n_hidden)

leftOutput = sharedLstm(encodedLeft)
rightOutput = sharedLstm(encodedRight)

output = concatenate([leftOutput, rightOutput])
output = Dense(1, activation = 'sigmoid')(output)

siameseLSTM = Model([leftInput, rightInput], [output])
optimizer = Adadelta(clipnorm=gradientClippingNorm)

siameseLSTM.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

training_start_time = time()

print("Started training")
for i in range(n_epoch):
	siameseLSTMTrained = siameseLSTM.fit([xTrain[0], xTrain[1]], yTrain.values, batch_size=batch_size, epochs=4,
                            	validation_data=([xValidation[0], xValidation[1]], yValidation.values))
	
	siameseLSTM_JSON = siameseLSTM.to_json()
	with open("../models/siameseLSTM_JSON.json","w") as json_file:
		json_file.write(siameseLSTM_JSON)
	siameseLSTM.save_weights("../models/siameseLSTM_WEIGHTS.h5")
	siameseLSTM.save("../models/siameseLSTM_model.h5")
	#simaeseLSTM = load_model(COMPUTE_DATA_PATH + 'siameseLSTM.h5')
	print("Finished epochs "+ repr(i+1))

print("Training time finished.\n{} epochs in {}".format(n_epoch, datetime.timedelta(seconds=time()-training_start_time)))

xTrain = [np.array(test_df['question1'].tolist()), np.array(test_df['question2'].tolist())]

for dataTuple in [xTrain]:
	for i in range(2):
		dataTuple[i] = pad_sequences(dataTuple[i], maxlen=maxSeqLength)

# load json and create model
json_file = open(MODELS_PATH + 'siameseLSTM_JSON.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(MODELS_PATH + "siameseLSTM_WEIGHTS.h5")

print("Loaded model from disk")

predictions = loaded_model.predict([xTrain[0],xTrain[1]])
print("predictions ready")
print("Geerating sub file")
import pandas as pdn

sub_df = pd.DataFrame(data=predictions,columns={"is_duplicate"})
sub_df.to_csv(path_or_buf="../results/kaggle_sub.csv", columns={"is_duplicate"}, header=True, index=True, index_label="test_id")

