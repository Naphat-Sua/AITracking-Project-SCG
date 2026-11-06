# Visit nectec website: https://www.nectec.or.th/corpus/index.php?league=pm
# and download these files:
# 1) article.zip
# 2) encyclopedia.zip
# 3) news.zip
# 4) novel.zip
#
# for testing
# 5) TEST_100K.txt
#
# The word-frequency list comes from TNC2_freq-5000.xls
# (download at: http://www.arts.chula.ac.th/~ling/TNC/category.php?id=58&)
# transferred to "Frequency.csv" in this folder.

import os
import pickle
import re
from os import listdir
from os.path import join

import numpy as np
import pandas as pd

PICKLE_FOLDER = "pickle"

# List of the words found most frequently in Thai language
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_FREQ_WORD = os.path.join(_THIS_DIR, "TNC2_freq-5000.csv")
if not os.path.exists(FILE_FREQ_WORD):
    # the copy committed in this repository
    FILE_FREQ_WORD = os.path.join(_THIS_DIR, "Frequency.csv")

_dict_word = None  # lazy cache of word -> index


def get_word_index():
    """Load (and cache) the word -> index dictionary from the frequency CSV."""
    global _dict_word
    if _dict_word is None:
        df_word2index = pd.read_csv(FILE_FREQ_WORD, index_col='word', usecols=['word', 'index'])
        _dict_word = df_word2index.to_dict(orient='dict')['index']
    return _dict_word


def get_index(word_list):
    """Convert words into indexes (using a dict lookup, so it is fast)."""
    dict_word = get_word_index()
    word_seqIndex = []
    for word in word_list:
        # select word found in the frequency list and get its index
        if word in dict_word:
            index = dict_word[word]
        # otherwise
        elif re.search('<NE>.*</NE>', word) is not None:  # for Named Entity
            index = 5001
        elif re.search('<AB>.*</AB>', word) is not None:  # for Abbreviation
            index = 5002
        elif re.search('<POEM>.*</POEM>', word) is not None:  # for Poem
            index = 5003
        else:  # rarely found
            index = 0
        word_seqIndex.append(index)
    return word_seqIndex


def _ensure_pickle_folder():
    os.makedirs(PICKLE_FOLDER, exist_ok=True)


def dataset2index(source_dir, pickle_file=None):
    # get all file names
    all_fileNames = [join(source_dir, file) for file in listdir(source_dir)]
    contentList = []

    for file in all_fileNames:
        with open(file, "r", encoding="utf8") as f:
            word_list = f.read().split('|')
            word_seqIndex = get_index(word_list)  # convert words into the index
        contentList.append(word_seqIndex)

    print("Total content: ", np.shape(contentList))  # shape output is (number files, )
    assert np.shape(contentList)[0] == len(all_fileNames)

    if pickle_file is not None:
        _ensure_pickle_folder()
        # save to a pickle file
        with open(join(PICKLE_FOLDER, pickle_file), "wb") as f:
            pickle.dump(contentList, f)
        # example how to load a pickle file:
        # contentList = pickle.load(open("article_thai.p", "rb"))
    return contentList


def content2index(file_name):
    import deepcut  # heavy import: only needed here

    with open(file_name, "r", encoding="utf8") as f:
        sentences = f.read().split(" ")  # split "space" char
    contentIndex = []
    for s in sentences:
        # tokenize thai words
        word_list = deepcut.tokenize(s)
        # and then convert words to indexes
        word_seqIndex = get_index(word_list)
        contentIndex = contentIndex + word_seqIndex
    return contentIndex


def create_classify_dataset(dataset_list):
    X_trainList, X_testList, Y_trainList, Y_testList = [], [], [], []
    for index, X in enumerate(dataset_list):
        total_example = len(X)
        # label 0: "article", label 1: "encyclopedia", label 2: "news", label 3: "novel"
        Y_label = [index] * total_example  # create label followed by index (class name)

        split_index = int(total_example * 0.80)
        X_train, X_test = X[0:split_index], X[split_index:]
        Y_train, Y_test = Y_label[0:split_index], Y_label[split_index:]

        X_trainList = X_trainList + X_train
        X_testList = X_testList + X_test
        Y_trainList = Y_trainList + Y_train
        Y_testList = Y_testList + Y_test

    return X_trainList, X_testList, Y_trainList, Y_testList


def load_dataset():
    with open(join(PICKLE_FOLDER, "classify_text.p"), "rb") as f:
        data = pickle.load(f)
    return (data['X_trainList'], data['X_testList'],
            data['Y_trainList'], data['Y_testList'])


def load_dataset_unknown(pickle_file):
    with open(join(PICKLE_FOLDER, pickle_file), "rb") as f:
        return pickle.load(f)


def main():
    output_path = ''
    # You can change this path for your datasets (downloaded from NECTEC, see header)
    SOURCE_PATH = os.environ.get("THAI_DATASET_PATH", "dataset-thai-word")

    _ensure_pickle_folder()

    print("\n------- Convert 'article' dataset to set of indexes ----------")
    article = dataset2index(join(SOURCE_PATH, "article"), join(output_path, "article_thai.p"))

    print("\n------- Convert 'encyclopedia' dataset to set of indexes ----------")
    encyclopedia = dataset2index(join(SOURCE_PATH, "encyclopedia"), join(output_path, "encyclopedia_thai.p"))

    print("\n------- Convert 'news' dataset to set of indexes ----------")
    news = dataset2index(join(SOURCE_PATH, "news"), join(output_path, "news_thai.p"))

    print("\n------- Convert 'novel' dataset to set of indexes ----------")
    novel = dataset2index(join(SOURCE_PATH, "novel"), join(output_path, "novel_thai.p"))

    print("\n------------ Create dataset for classify task ----------")
    dataset_list = [article, encyclopedia, news, novel]
    X_trainList, X_testList, Y_trainList, Y_testList = create_classify_dataset(dataset_list)
    assert len(X_trainList) + len(X_testList) == len(Y_trainList) + len(Y_testList)
    assert len(X_trainList) + len(X_testList) == len(article) + len(encyclopedia) + len(news) + len(novel)

    data = {
        'X_trainList': X_trainList,
        'X_testList': X_testList,
        'Y_trainList': Y_trainList,
        'Y_testList': Y_testList,
    }
    # save dictionary to a pickle file
    with open(join(PICKLE_FOLDER, "classify_text.p"), "wb") as f:
        pickle.dump(data, f)

    print("\n------------ Convert unknown dataset (for test) to set of indexes ----------")
    file_name = "Novel.txt"  # a sample Thai novel text committed in this folder
    contentList = content2index(file_name)
    # save to a pickle file
    with open(join(PICKLE_FOLDER, file_name + ".p"), "wb") as f:
        pickle.dump(contentList, f)


if __name__ == "__main__":
    main()
