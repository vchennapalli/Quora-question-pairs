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

del train_df
inverse_dictionary = np.load(COMPUTE_DATA_PATH + 'inverse_dictionary.npy').item()
for key, value in inverse_dictionary.items():
	inverse_dictionary[key] = value.encode('ascii')


dictionary = {}
maxSeqLength = 100

for index in range(len(inverse_dictionary)):
	dictionary[inverse_dictionary[index].decode("utf-8")] = index
	
for dataTuple in [test_df]:
	for index, row in dataTuple.iterrows():
		for question in question_cols:
			numVector = []
			for word in question_to_wordlist(row[question]):
				if (word in dictionary):
					numVector.append(dictionary[word])
			if (len(numVector) > 100):
				numVector = numVector[0:100]
			dataTuple.set_value(index, question, numVector)

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

