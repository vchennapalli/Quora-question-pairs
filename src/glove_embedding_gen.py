from glove import Glove
from glove import Corpus

from sentenceToWordList import *
import numpy as np

def read_data(filenames):
    """
    input - filenames
    output - a list of words in the question
    """
    for f in filenames:
        for i, r in f.iterrows():
            for q in question_cols:
                yield question_to_wordlist(r[q])


filenames = [train_df, test_df]
#filenames = [train_df]
print("Preprocessing corpus")
get_data = read_data
corpus_model = Corpus()
corpus_model.fit(get_data(filenames), window=10)

#corpus_model.save(COMPUTE_DATA_PATH + '/corpus.model')

print('Dict size: %s' % len(corpus_model.dictionary))
print('Collocations: %s' % corpus_model.matrix.nnz)
#corpus_model = Corpus.load(COMPUTE_DATA_PATH + '/corpus.model')

glove = Glove(no_components=300, learning_rate=0.05)
print("Starting training")
glove.fit(corpus_model.matrix, epochs=1000,
          no_threads=6, verbose=True)

#glove = Glove.load(COMPUTE_DATA_PATH + '/glove.model')

glove.add_dictionary(corpus_model.dictionary)

np.savetxt(COMPUTE_DATA_PATH + 'g_embedding_matrix.txt', glove.word_vectors, fmt = "%.5f")

np.save(COMPUTE_DATA_PATH + 'g_inverse_dictionary.npy', glove.inverse_dictionary)
np.save(COMPUTE_DATA_PATH + 'g_dictionary.npy', glove.dictionary)


#glove.save(COMPUTE_DATA_PATH + '/glove.model')           
