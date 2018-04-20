"""
Uses pre-trained GloVe based word-embedding to develop needed embedding matrix.
This could be found at http://nlp.stanford.edu/data/glove.6B.zip
Utilized this file to calculate the potential typos and shorthands for words
REQUIREMENT: Needs the positioning of the file as given in EMBEDDINGS_FILE variable
"""

import numpy as np
from sentenceToWordList import *

EMBEDDINGS_FILE = "../../pretrained_embeddings/glove.6B.50d.txt"
embeddingVectorSize = 50
NON_DICT_WORDS_FILE = "../processed_data/non_dict_words.txt"

"""creates the collection of all the words in the datasets and returns """
def readData(files):
    """
    input: files - list
    output: all the word from the train and test datafiles - list
    """
    vocab = []
    for f in files:
        for i, r in f.iterrows():
            for q in question_cols:
                vocab.extend(question_to_wordlist(r[q]))
    return vocab

def createEmbeddingMatrix(words):
    """
    output: a dict object having mapping from word index to the embedding
    """
    print("GERE")
    size = 0
    inBoth = set()
    with open(EMBEDDINGS_FILE) as f:
        for line in f:
            wordcoeffs = line.split()
            word = wordcoeffs[0]
            if word in words:
                inBoth.add(word)
                size += 1
    err = list(words.difference(inBoth))
    with open(NON_DICT_WORDS_FILE, 'w') as f:
        for item in err:
            f.write("%s\n" % item)

def createDictionary(files):
    """
    input: files for which number to word mapping has to be created.
    output: returns dictionary and inverse dictionary of the Quora dataset.
    """
    vocab = readData(files)
    words = set(vocab)
    dictionary = dict()
    for word in words:
        dictionary[word] = len(dictionary)
    inverseDict = dict(zip(dictionary.values(), dictionary.keys()))
    return words, dictionary, inverseDict

files = [train_df, test_df]
#files = [train_df]
words, dictionary, inverseDict = createDictionary(files)
createEmbeddingMatrix(words)
#np.savetxt(COMPUTE_DATA_PATH + 'glove_embedding_matrix.txt', embeddingMatrix, fmt = "%.5f")
#np.save(COMPUTE_DATA_PATH + 'glove_inverse_dictionary.npy', inverseDict)
#np.save(COMPUTE_DATA_PATH + 'glove_dictionary.npy', dictionary)
