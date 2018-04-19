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
		

for dataTuple in [train_df, test_df]:
	for index, row in dataTuple.iterrows():
		a = nodes[row['question1']]
		b = nodes[row['question2']]
		dataTuple.set_value(index, 'qid1', a)
		dataTuple.set_value(index, 'qid2', b)
		dataTuple.set_value(index, 'min_freq', min(freq[a], freq[b]))
		dataTuple.set_value(index, 'common_neighbours', len(edges[a].intersection(edges[b])))

train_df.to_csv(TRAIN_CSV,sep=',',index=False)
test_df.to_csv(TEST_CSV,sep=',',index=False)
