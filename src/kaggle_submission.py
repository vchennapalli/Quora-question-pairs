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
from keras.models import model_from_json

import urllib
import os

import numpy as np
import tensorflow as tf
from sentenceToWordList import *

inverse_dictionary = np.load(COMPUTE_DATA_PATH + 'inverse_dictionary.npy').item()
for key, value in inverse_dictionary.items():
	inverse_dictionary[key] = value.encode('ascii')


dictionary = {}
maxSeqLength = 37

for index in range(len(inverse_dictionary)):
	dictionary[inverse_dictionary[index].decode("utf-8")] = index
	
for dataTuple in [test_df]:
	for index, row in dataTuple.iterrows():
		for question in question_cols:
			numVector = []
			for word in question_to_wordlist(row[question]):
				if (word in dictionary):
					numVector.append(dictionary[word])
			dataTuple.set_value(index, question, numVector)
			maxSeqLength = max(maxSeqLength, len(numVector))

validation_size = 0
xTrain, xValidation, yTrain, yValidation = train_test_split(test_df[question_cols], test_df['test_id'], test_size=validation_size)

xTrain = [xTrain.question1, xTrain.question2]
xValidation = [xValidation.question1, xValidation.question2]
for dataTuple in [xTrain, xValidation]:
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

data_sub = {'test_id':yTrain, 'is_duplicate': predictions}

sub_df = pd.DataFrame(data=data_sub, columns={'test_id','is_duplicate'})
sub_df = sub_df[['test_id','is_duplicate']]
sub_df.to_csv(path_or_buf=RESULTS_PATH + "kaggle_sub.csv",columns={"test_id","is_duplicate"},header=True)
print("File ready!")
