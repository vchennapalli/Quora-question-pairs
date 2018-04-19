import re
import pandas as pd

#FILEPATHS
TEST_CSV1 = "../processed_data/split_p_test00.csv"
TEST_CSV2 = "../processed_data/split_p_test01.csv"
TRAIN_CSV = "../processed_data/p_train.csv"
COMPUTE_DATA_PATH = "../computed_data/"
MODELS_PATH = "../models/"
RESULTS_PATH = "../results/"
PROCESSED_DATA_PATH = "../processed_data/"

#LOADS TRAINING AND TEST SET
train_df = pd.read_csv(TRAIN_CSV)
test_df1 = pd.read_csv(TEST_CSV1)
test_df2 = pd.read_csv(TEST_CSV2)
test_df = pd.concat([test_df1, test_df2])

print(test_df.shape)
print(train_df.shape)

vocab_size = 121326 #150
question_cols = ['question1', 'question2']
maxSeqLength = 65

"""preprocesses and converts question to a list of words"""
def question_to_wordlist(text):
    """
    input: string of text
    output: list of words
    """
    text = str(text)
    text = text.lower()

    #text cleaning and substitutions
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.split()

    return text
