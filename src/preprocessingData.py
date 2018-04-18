from textblob import TextBlob
import pandas as pd
import re
from textblob import Word

TRAIN_CSV = "~/raw_data/train.csv"
TEST_CSV = "~/raw_data/test.csv"
PROCESSED_DATA_PATH = "../processed_data/"

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

def process(text, translation):
    for token, replacement in translation.items():
        text = text.replace(token, ' ' + replacement + ' ')
    text = text.replace('  ', ' ')
    return text

def digits_to_text(text):
    translation = {
        '0': 'zero',
        '1': 'one',
        '2': 'two',
        '3': 'three',
        '4': 'four',
        '5': 'five',
        '6': 'six',
        '7': 'seven',
        '8': 'eight',
        '9': 'nine',
    }
    return process(text, translation)

def spell_correct(text):
    text = TextBlob(text)
    nouns = text.noun_phrases
    nouns = ' '.join(nouns)
    nouns = nouns.split()
    words = text.lower().words
        
    output = []
    for word in words:
        if word in nouns:
            output.append(word)
        else:
            output.append(str(Word(word).correct()))
    string_question = ' '.join(str(word) for word in output)
    return string_question

def negation_translate(text):
    translation = {
        "can't": 'can not',
        "won't": 'would not',
        "shan't": 'shall not',
    }
    text = process(text, translation)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=\s]", "",text)
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
    
    return text

def process_question(question, spellcheck=True):
    if spellcheck:
        question = spell_correct(question)
    
    question = digits_to_text(question)
    question = negation_translate(question)

    return question

#remove id, qid1, qid2
def process_dataFrame(df):
    df['question1'] = df.apply(lambda row: process_question(row['question1']),axis=1)
    df['question2'] = df.apply(lambda row: process_question(row['question2']),axis=1)


#process both test and train data frames
process_dataFrame(train_df)
print("Train Processing done")
process_dataFrame(test_df)
print("Test processing done")

#min_freq
#common_neighbours
#qid1, qid2
def add_cols_dataframe(df):
    df['min_freq'] = 0
    df['common_neighbours'] = 0

def add_qids_dataframe(df):
    len = df.shape[0]
    evens = [x for x in range(len*2+2) if x%2 == 0 and x !=0]
    odds = [x for x in range(len*2) if x%2 != 0]
    df['qid1'] = odds
    df['qid2'] = evens

#add cols for both test and train data frames
add_cols_dataframe(train_df)
add_cols_dataframe(test_df)

#add for only test data frame
add_qids_dataframe(test_df)

#save dataframes as csv
train_df.to_csv(PROCESSED_DATA_PATH + "p_train.csv",sep=',',index=False)
test_df.to_csv(PROCESSED_DATA_PATH + "p_test.csv",sep=',',index=False)
print("processed files saved")

