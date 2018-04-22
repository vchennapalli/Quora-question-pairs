import itertools

import urllib
import os

import numpy as np
from sentenceToWordList import *

from fuzzywuzzy import fuzz

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
        dataTuple['fuzzy_qratio'] = 0
        dataTuple['fuzzy_wratio'] = 0
        dataTuple['fuzzy_partial_ratio'] = 0
        dataTuple['fuzzy_partial_token_set_ratio'] = 0
        dataTuple['fuzzy_partial_token_sort_ratio'] = 0
        dataTuple['fuzzy_token_set_ratio'] = 0
        dataTuple['fuzzy_token_sort_ratio'] = 0
			
        for index, row in dataTuple.iterrows():	
                a = nodes[row['question1']]
                b = nodes[row['question2']]
                dataTuple.set_value(index, 'qid1', a)
                dataTuple.set_value(index, 'qid2', b)
	        	
                dataTuple.set_value(index, 'min_freq', min(freq[a], freq[b]))
                dataTuple.set_value(index, 'common_neighbours', len(edges[a].intersection(edges[b])))
              
                q1, q2 = str(row['question1']), str(row['question2'])
                q_len1, q_len2 = len(q1), len(q2)

                #question length metrics
                dataTuple.set_value(index, 'q_len1', q_len1)
                dataTuple.set_value(index, 'q_len2', q_len2)
                dataTuple.set_value(index, 'diff_len', q_len1 - q_len2)
          
                words_q1, words_q2 = q1.split(), q2.split()
                word_len1, word_len2 = len(words_q1), len(words_q2)

                #number of words metric
                dataTuple.set_value(index, 'word_len1', word_len1)
                dataTuple.set_value(index, 'word_len2', word_len2)
                tmp = len(set(words_q1).intersection(words_q2))
                dataTuple.set_value(index, 'common_words', (tmp))

                #fuzzy metrics
                dataTuple.set_value(index, 'fuzzy_qratio', fuzz.QRatio(q1, q2))
                dataTuple.set_value(index, 'fuzzy_wratio', fuzz.WRatio(q1, q2))
                dataTuple.set_value(index, 'fuzzy_partial_ratio', fuzz.partial_ratio(q1, q2))
                dataTuple.set_value(index, 'fuzzy_partial_token_set_ratio', fuzz.partial_token_set_ratio(q1, q2))
                dataTuple.set_value(index, 'fuzzy_partial_token_sort_ratio', fuzz.partial_token_sort_ratio(q1, q2))
                dataTuple.set_value(index, 'fuzzy_token_set_ratio', fuzz.token_set_ratio(q1, q2))
                dataTuple.set_value(index, 'fuzzy_token_sort_ratio', fuzz.token_sort_ratio(q1, q2))              
              

                if (index%10000 == 0):
                    print(index)

 
train_df.to_csv(TRAIN_CSV,sep=',',index=False)
test_df.to_csv(TEST_CSV,sep=',',index=False)
