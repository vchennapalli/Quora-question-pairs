import itertools

import urllib
import os

import numpy as np
from sentenceToWordList import *

nodes = {}
nodeCount = 0
freq = {}
edges = {}

for dataTuple in [train_df, test_df]:
	for index, row in dataTuple.iterrows():
		nodeIds = []
		for question in question_cols:
			if (row[question] not in nodes):
				nodes[row[question]] = nodeCount
				freq[nodeCount] = 0
				edges[nodeCount] = set()
				nodeCount = nodeCount + 1
			tmpNodeId = nodes[row[question]]
			freq[tmpNodeId] = freq[tmpNodeId] + 1
			nodeIds.append(tmpNodeId)
		edges[nodeIds[0]].add(nodeIds[1])
		edges[nodeIds[1]].add(nodeIds[0])
		if (index%10000 == 0):
			print(index)

for dataTuple in [train_df, test_df]:
	dataTuple['q_len1'] = 0
	dataTuple['q_len2'] = 0
	dataTuple['diff_len'] = 0
	dataTuple['word_len1'] = 0
	dataTuple['word_len2'] = 0
	dataTuple['common_words'] = 0
			
	for index, row in dataTuple.iterrows():
		a = nodes[row['question1']]
		b = nodes[row['question2']]
		dataTuple.set_value(index, 'qid1', a)
		dataTuple.set_value(index, 'qid2', b)
		
		dataTuple.set_value(index, 'min_freq', min(freq[a], freq[b]))
		dataTuple.set_value(index, 'common_neighbours', len(edges[a].intersection(edges[b])))
		dataTuple.set_value(index, 'q_len1', len(str(row['question1'])))
		dataTuple.set_value(index, 'q_len2', len(str(row['question2'])))
		dataTuple.set_value(index, 'diff_len', len(str(row['question2'])) - len(str(row['question1'])))
		dataTuple.set_value(index, 'word_len1', len(str(row['question1']).split()))
		dataTuple.set_value(index, 'word_len2', len(str(row['question2']).split()))
		tmp = len(set(str(row['question1']).split()).intersection(str(row['question2']).split()))
		dataTuple.set_value(index, 'common_words', (tmp))
		if (index%10000 == 0):
			print(index)

 
train_df.to_csv(TRAIN_CSV,sep=',',index=False)
test_df.to_csv(TEST_CSV,sep=',',index=False)
