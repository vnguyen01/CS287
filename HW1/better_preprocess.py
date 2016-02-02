"""Text Classification Preprocessing
"""

import numpy as np
import h5py
import argparse
import sys
import re
from math import log

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier

from collections import Counter


# Different data sets to try.
# Note: TREC has no development set.
# And SUBJ and MPQA have no splits (must use cross-validation)
FILE_PATHS = {"SST1": ("data/stsa.fine.phrases.train",
                       "data/stsa.fine.dev",
                       "data/stsa.fine.test"),
              "SST2": ("data/stsa.binary.phrases.train",
                       "data/stsa.binary.dev",
                       "data/stsa.binary.test"),
              "TREC": ("data/TREC.train.all", None,
                       "data/TREC.test.all"),
              "SUBJ": ("data/subj.all", None, None),
              "MPQA": ("data/mpqa.all", None, None)}

args = {}

def line_to_words(line, dataset):
    # Different preprocessing is used for these datasets.
    if dataset in ['SST1', 'SST2']:
        clean_line = clean_str_sst(line.strip())
    else:
        clean_line = clean_str(line.strip())
    words = clean_line.split(' ')
    words = words[1:]
    return words


def get_vocab(file_list, dataset=''):
    """
    Construct index feature dictionary.
    EXTENSION: Change to allow for other word features, or bigrams.
    """
    file_list_examples = {}
    word_to_idx = {}

    num_docs = 0
    doc_freq = {}

    # Start at 2 (1 is padding)
    idx = 0
    for filename in file_list:
        file_list_examples[filename] = 0
        if filename:
            with open(filename, "r") as f:
                for line in f:
                    num_docs += 1
                    file_list_examples[filename] += 1
                    words = line_to_words(line, dataset)
                    for word in words:
                        if word in doc_freq:
                            doc_freq[word] += 1
                        else:
                            doc_freq[word] = 1
                        if word not in word_to_idx:
                            word_to_idx[word] = idx
                            idx += 1

    return file_list_examples, word_to_idx, num_docs, doc_freq

def get_idf(word, num_docs, doc_freq):
    if not word in doc_freq:
        return 1.5
    idf = float(num_docs)/doc_freq[word]
    return log(idf, 2)

def convert_data(data_name, word_to_idx, file_list_examples, num_docs, doc_freq, dataset):
    """
    Convert data to padded word index features.
    EXTENSION: Change to allow for other word features, or bigrams.
    """

    lbl = []

    doc_word = np.zeros(shape=(file_list_examples[data_name], len(word_to_idx)))

    with open(data_name, 'r') as f:
        for i, line in enumerate(f):

            words = line_to_words(line, dataset)
            y = int(line[0]) + 1

            """
            TFIDF 

            #count weights
            words_set = set(words)
            tfidf = {}

            for word in words_set:
                tf = float(words.count(word))/len(words)
                idf = get_idf(word, num_docs, doc_freq)
                tfidf[word] = tf*idf
            
            s = (sum(map(lambda x: x ** 2, tfidf.values())) **.5)
            tfidf = {word: v / s for word,v in tfidf.iteritems()}

            for word in tfidf:
                doc_word[i][word_to_idx[word]] = tfidf[word]
            """

            """
            COUNT
            """
            for word in words:
                doc_word[i][word_to_idx[word]] += 1
            

            """
            BINARY 
            
            for word in set(words):
                doc_word[i][word_to_idx[word]] = 1
            """
            lbl.append(y)

    return doc_word, np.array(lbl, dtype=np.int32)


def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', help="Data set",
                        type=str)
    args = parser.parse_args(arguments)
    dataset = args.dataset
    train, valid, test = FILE_PATHS[dataset]

    

    # Features are just the words.
    file_list_examples, word_to_idx, num_docs, doc_freq = get_vocab([train, valid, test], dataset)

    # Dataset name
    train_input, train_output = convert_data(train, word_to_idx, file_list_examples, num_docs, doc_freq,
                                             dataset)

    print train_input.shape
    print train_output.shape



    if valid:
        valid_input, valid_output = convert_data(valid, word_to_idx, file_list_examples, num_docs, doc_freq,
                                                 dataset)
        print valid_input.shape
        print valid_output.shape

    if test:
        test_input, _ = convert_data(test, word_to_idx, file_list_examples, num_docs, doc_freq,
                                 dataset)
        print test_input.shape

    V = len(word_to_idx) + 1
    print('Vocab size:', V)


    C = np.max(train_output)
    print Counter(train_output)
    print Counter(valid_output)

    clf = RandomForestClassifier()
    clf.fit(train_input, train_output)
    print clf.score(valid_input, valid_output)
    
    """
    filename = args.dataset + '.hdf5'
    with h5py.File(filename, "w") as f:
        f['train_input'] = train_input
        f['train_output'] = train_output
        if valid:
            f['valid_input'] = valid_input
            f['valid_output'] = valid_output
        if test:
            f['test_input'] = test_input
        f['nfeatures'] = np.array([V], dtype=np.int32)
        f['nclasses'] = np.array([C], dtype=np.int32)
    """
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
    main(args)