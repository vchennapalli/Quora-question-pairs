import itertools

from time import time
from nltk.corpus import stopwords


import numpy as np
from sentenceToWordList import *

#del test_df
stops = set(stopwords.words('english'))
maxSeqLength = 0

lengths = []
redLen = []
for dataTuple in [train_df, test_df]:
	for index, row in dataTuple.iterrows():
		for question in question_cols:
			numVector = 0
			for word in question_to_wordlist(row[question]):
					numVector = numVector + 1
			lengths.append(numVector)
			if (numVector > 65):
				numVector = 0
				for word in question_to_wordlist(row[question]):
					if (word not in stops):
						numVector = numVector + 1
			redLen.append(numVector)
			if (maxSeqLength < numVector):
				maxSeqLength = numVector

lengths = np.array(lengths)
print(np.percentile(lengths, 99.6))
print(np.percentile(lengths, 99.7))
print(np.percentile(lengths, 99.8))
print(np.percentile(lengths, 99.9))
print(np.percentile(lengths, 99.93))
print(np.percentile(lengths, 99.95))
print(np.percentile(lengths, 99.96))
print(np.percentile(lengths, 99.97))
print(np.percentile(lengths, 99.98))
print(np.percentile(lengths, 99.99))

print("maxSeqLEngth")
print(maxSeqLength)


print(np.percentile(redLen, 99.6))
print(np.percentile(redLen, 99.7))
print(np.percentile(redLen, 99.8))
print(np.percentile(redLen, 99.9))
print(np.percentile(redLen, 99.93))
print(np.percentile(redLen, 99.95))
print(np.percentile(redLen, 99.96))
print(np.percentile(redLen, 99.97))
print(np.percentile(redLen, 99.98))
print(np.percentile(redLen, 99.99))
