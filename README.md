## Detecting semantically identical questions pairs using Deep Learning

## Problem Statement
A pair of questions are said to be identical if they have the same intent. Identifying these kind of semantically identical question pairs has been a crucial challenge in the field of Natural Language Processing (NLP). Obtaining an accurate solution for this would benefit the users of question and answer forum based websites such as Quora, stack overflow, reddit etc. Furthermore, the solutions obtained here could be used to solve other problems that are central to the field of NLP.

We propose an efficient model to identify semantically identical question pairs by making use of Siamese Long Short Term Memory Networks (LSTMN) model. We made use of Glove, word2vec skip gram paradigm along with negative sampling to develop and train the embedding matrix. For the main model, we made use of concatenation to predict the similarity initially later experimented with various distance measures like Manhattan, Euclidean etc. 


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
|  |  |  |
|--|--|--|
| anaconda | custom | py27h4a00acb_0|                             
|conda|                     4.3.30|           py27h6ae6dc7_0|
|Fuzzywuzzy     |           0.16.0     |               pip|
|h5py           |           2.7.0  |             np113py27_0|
|hdf5         |             1.8.17 |                       2|
|jupyter      |             1.0.0         |            pip|
|Keras      |               2.0.9         |            pip|
|matplotlib   |             2.0.2      |         np113py27_0|
|numpy        |             1.13.3      |              pip|
|NLTK				|3.2.5							| pip|
|pandas              |      0.21.0           |        pip|
|pip           |            8.1.2      |           py27_0|
|python                 |   2.7.11         |            5|
|scikit-learn          |    0.19.1        | py27h445a80a_0|
|scipy          |           1.0.0           |          pip|
|TextBlob       |           0.15.1      |              pip|
|tensorflow-gpu   |         1.4.0     |                pip|
|tensorflow-tensorboard  |  0.4.0rc2  |                pip|
|xgboost           |        0.6        |               pip|

### Requirements
- Dataset: The dataset can be downloaded from the [Kaggle competition](https://www.kaggle.com/c/quora-question-pairs/data).
- The raw data downloaded should be placed in the `raw_data` directory inside project root folder.
- Libraries Used: numpy, pandas, scikit-learn, matplot-lib, keras, tensorflow
- Toolkit Used: nlkt 

### Project Directory
* Quora-question-pairs
   - computed_data - *To save the computed embeddings, dictionary and inverse dictionary*
   - models - *To save thetrained models* 
   - processed_data - *To save the initial processed data - spell corrections, digit translations, cleaning, expand negations, lower case etc*
   - src - *contains the python files to process, train and get predictions from the model* 
   - data_paths.txt - *holds the paths for the required files*
   
## Architecture
![Architecture](images/arc_dense.png?raw=true "Architecture of the implemented model")

Tried with the concatenation of LSTM outputs passed through a dense layer as well as other similarity functions like Manhattan distance, Euclidean Distance, Cosine Similarity etc.

## Steps to run the code

* git clone https://github.com/vchennapalli/Quora-question-pairs.git

* cd Quora-question-pairs/

* unzip computed_data/g_embedding_matrix.zip processed_data/processed_test_00.zip processed_data/processed_test_01.zip processed_data/processed_train.zip

* cat processed_data/split_p_test00.csv processed_data/split_p_test01.csv > processed_data/p_test.csv

* python3 siameseNetwork.py     // Trains the model and generates predictions.

* unzip models/siameseLSTM_MODEL.zip

* python3 kaggle_submission.py  // For generating predictions on an already trained model.



## Test results
* First run with basic implementation
  - Trend of Accuracy and Log Loss percentage over number of epochs on train data set
  ![Train Basic Results](images/train.png?raw=true "Test results trend")
  - Trend of Accuracy and Log Loss percentage over number of epochs on validation data set
  ![Cross Validation Basic Results](images/validation.png?raw=true "Validation results trend")

* | Model | Validation Accuracy  | Public Log loss on Kaggle |
|--|--|-- |
| Siamese LSTM + custom trained word2vec embeddings|76  | 0.39243 |
| Siamese LSTM + word2vec pretrained embeddings|75  | 0.43440 |
| Siamese LSTM + custom trained Glove embeddings|77  | NA |
| Siamese LSTM + Glove + Magic features | 83 | 0.32236|
| Siamese LSTM + Glove + Magic features + Normal features | 86 | 0.26116|
| Siamese LSTM + Glove + Magic features + Normal features + Class weights| 82 | 0.23826|
| Siamese LSTM + Glove + Magic features  + Normal features + Class weights + Stacked layers| 83 | 0.20457|

* On Kaggle competition

## Built With

* [Glove Embeddings](https://nlp.stanford.edu/projects/glove/) - An unsupervised learning algorithm for obtaining vector representations for words
* [Keras](https://keras.io/) - High-level neural networks API - written in Python
* [TensorFlow ](https://www.tensorflow.org/) - An open-source software library for dataflow programming across a range of tasks
* [FuzzyWuzzy](https://pypi.org/project/fuzzywuzzy/) - Fuzzy string matching using Levenshtein Distance

## Authors

* **Abhinav Reddy Podduturi** - [Graduate student at Univeristy of Florida](https://github.com/Abhinav-Reddy)
* **Chanikya Mohan** - [Graduate student at Univeristy of Florida](https://github.com/ChanikyaMohan)
* **Vineeth Chennapalli** - [Graduate student at Univeristy of Florida](https://github.com/vchennapalli)


## References

[1] Jonas Mueller, Aditya Thyagarajan. (2016). “Siamese Recurrent Architectures for Learning Sentence Similarity" [Paper]. Retrieved from [https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12195/12023](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12195/12023)

[2] Tomas Mikolov, Kai Chen, Greg Corrodo, Jeffrey Dean. (2013). “Efficient Estimation of Word Representations in Vector Space” [Paper]. Retrieved from [https://arxiv.org/pdf/1301.3781.pdf](https://arxiv.org/pdf/1301.3781.pdf)

[3] Tomas Mikolov, Kai Chen, Greg Corrodo, Jeffrey Dean, Ilya Sutskever. (2013). “Distributed Representations of Words and Phrases and their Compositionality” [Paper]. Retrieved from [https://arxiv.org/pdf/1310.4546.pdf](https://arxiv.org/pdf/1310.4546.pdf).

[4] Sepp Hochreiter and Jurgen Schmidhuber. (1997). “Long Short Term Memory”. Neural Computation 9(8): 1735-1780

[5] Elkhan Dadashov, Sukolsak Sakshuwong and Katherine Yu, “Quora Question Duplication”.

[6] Word2vec Keras Tutorial [Website] “[http://adventuresinmachinelearning.com/word2vec-keras-tutorial/](http://adventuresinmachinelearning.com/word2vec-keras-tutorial/)”

[7]Word2vec Skip Gram Tutorial [Website]
“[http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)”

[8] How to predict Quora Question Pairs using Siamese Manhattan LSTM [Website] "https://medium.com/mlreview/implementing-malstm-on-kaggles-quora-question-pairs-competition-8b31b0b16a07"
