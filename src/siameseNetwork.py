from keras.models import Model
from keras.layers import Input, Dense, Reshape, merge
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import skipgrams
from nltk.corpus import stopwords
from keras.preprocessing import sequence

import urllib
import os

import numpy as np
import tensorflow as tf
from sentenceToWordList import *

inverse_dictionary = np.loadtxt(COMPUTE_DATA_PATH + 'inverse_dictionary.npy')
dictionary = {}
maxSeqLength = max(test_df.question1.map(lambda x: len(x)).max(),
               test_df.question2.map(lambda x: len(x)).max())
del test_df

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

validation_size = 40000
xTrain, xValidation, yTrain, yValidation = train_test_split(train_df[questions_cols], train_df['is_duplicate'], test_size=validation_size)
for dataTuple in [xTrain, xTest]:
	for question in questions_cols:
		dataTuple[question] = pad_sequences(dataset[question], maxLen=maxSeqLength)


# Model variables
n_hidden = 50
gradientClippingNorm = 1.25
batch_size = 64
#keep n_epoch a multiple of 5
n_epoch = 50
embedding_dim = 300
embeddingsMatrix = np.loadtxt(COMPUTE_DATA_PATH + 'embedding_matrix.txt', fmt = "%.5f")


def exponent_neg_manhattan_distance(left, right):
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

leftInput = Input(shape=(maxSeqLength,), dtype='int32')
rightInput = Input(shape=(maxSeqLength,), dtype='int32')

embeddingLayer = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_length=maxSeqLength, trainable=False)

encodedLeft = embedding_layer(leftInput)
encodedRight = embedding_layer(rightInput)

sharedLstm = LSTM(n_hidden)

leftOutput = sharedLstm(encodedLeft)
rightOutput = sharedLstm(encodedRight)
distance = keras.layers.add()

output = keras.layers.concatenate([leftOutput, rightOutput])
output = Dense(1, activation = 'sigmoid')(output)
#malstmDistance = Merge(mode=lambda x: exponent_neg_manhattan_distance(x[0], x[1]), output_shape=lambda x: (x[0][0], 1))([leftOutput, rightOutput])

siameseLSTM = Model([leftInput, rightInput], [output])
optimizer = Adadelta(clipnorm=gradientClippingNorm)

siameseLSTM.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

training_start_time = time()

for i in range(n_epoch/5):
	siameseLSTMTrained = siameseLSTM.fit([xTrain['question1'], xTrain['question2']], yTrain.values, batch_size=batch_size, nb_epoch=n_epoch,
                            	validation_data=([xValidation['question1'], xValidation['question2']], yValidation.values))
	siameseLSTMTrained.save('siameseLSTMTrained.h5')

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
