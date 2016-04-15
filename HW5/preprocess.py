#!/usr/bin/env python

"""
NER Preprocessing
"""

import csv
from collections import Counter
import collections
import numpy as np
import operator
import h5py
import argparse
import sys
import re
import codecs
import string
import itertools
import gc
import copy

FILE_PATHS = {"full": ("data/train.num.txt",
                      "data/dev.num.txt",
                      "data/dev_kaggle.txt",
                      "data/tags.txt",
                      "data/test.num.txt",
                      "data/glove.txt")}

train_path, valid_path, valid_kaggle_path, tags_path, test_path, embeddings_path = FILE_PATHS["full"]

train = (list(csv.reader(open(train_path, 'rb'), delimiter='\t', quotechar='|')))
valid = (list(csv.reader(open(valid_path, 'rb'), delimiter='\t',quotechar='|')))
valid_kaggle = (list(csv.reader(open(valid_kaggle_path, 'rb'), delimiter='\t', quotechar='|')))
tags = (list(csv.reader(open(tags_path, 'rb'), delimiter=' ')))
test = (list(csv.reader(open(test_path, 'rb'), delimiter=' ', quotechar='|')))

dataset = [train, valid, test]

def is_float(c):
    if c.isdigit():
        return True
    if c == ',':
        return True
    return False

def word_process(word):
    #Realized that this can be done much easier with regular expresions...
    #Removes digits and replaces them with the string 'NUMBER'
    flag = 0
    interval = [0,0]
    intervals = []
    processed = ''
    if len(word) == 1 and (not word.isdigit()):
        return word
    for i in range(0,len(word)):
        if flag == 0:
            if is_float(word[i]):
                flag = 1
                interval[0] = i
                #if len(word) == 1:
                    #intervals.append((0,0))
                if i == len(word)-1:
                    intervals.append((interval[0], interval[0]+1))
        else:
            if (not is_float(word[i])):
                flag = 0
                interval[1] = i-1
                intervals.append((interval[0],interval[1]))
            elif i == len(word) -1:
                interval[1] = i
                intervals.append((interval[0],interval[1]))
                
    index = 0
    for pair in intervals:
        prefix = word[index:pair[0]]
        index = pair[1]+1
        processed = processed + prefix + 'NUMBER'
    
    processed = processed + word[index:]
    return processed      

def gen_dictionary(data_set, tag_set):
    d = collections.defaultdict(list)
    d['<s>'] = 1
    d['</s>'] = 2
    counter = 3
    for data in data_set:
        for element in data:
            if element != []:
                word = word_process(element[2])
                if word not in d:
                    d[word] = counter
                    counter = counter + 1
    for i in range(0, len(tag_set)):
        tag_set[i][1] = int(tag_set[i][1])
    tag_set = dict(tag_set)
    tag_set['<s>'] = 8
    tag_set['</s>'] = 9
    
    return d, tag_set

def lex_features(data, v_dict):
    lex = []
    for i in range(0,len(data)):
        row = data[i]
        if row != []:
            if int(row[1]) == 1:
                lex.append(v_dict['<s>'])
            lex.append(v_dict[word_process(row[2])])
    return np.array(lex)

def get_features(data, v_dict, t_dict, feature_functions):
    inputs = feature_functions[0](data, v_dict)
    for i in range(1, len(feature_functions)):
        inputs = np.vstack((inputs, feature_functions[0](data,v_dict)))
    return inputs
        

def get_outputs(data, t_dict):
    outputs = []
    for i in range(0, len(data)):
        row = data[i]
        if row != []:
            if int(row[1]) == 1:
                outputs.append(t_dict['<s>'])
            outputs.append(t_dict[row[3]])
    return np.array(outputs)

def max_len(data):
    maxlen = 0
    count = 0
    for i in range(0, len(data)):
        row = data[i]
        if row != []:
            maxlen = max(int(row[1]), maxlen)
            if int(row[1]) == 1:
                count = count + 1
    return maxlen, count

def get_sentences(X_data, maxlen, numsen):
    #takes in the index form of the X data, maximum sentence length, and number of sentences
    sentences = np.zeros((numsen, maxlen+1))
    sen_ind = -1
    word_ind = 0
    for ind in X_data:
        if ind == 1:
            sen_ind = sen_ind + 1
            word_ind = 0
        sentences[sen_ind, word_ind] = ind
        word_ind = word_ind + 1
    return sentences

def main():
    features = [lex_features]

    vocab_dict, tag_dict = gen_dictionary(dataset, tags)
    Y_train = get_outputs(train, tag_dict)
    Y_valid = get_outputs(valid, tag_dict)
    X_train = get_features(train, vocab_dict, tag_dict, features)
    X_valid = get_features(valid, vocab_dict, tag_dict, features)
    X_test = get_features(test, vocab_dict, tag_dict, features)

    #sentences
    max_length_valid, num_sentence_valid = max_len(valid)
    max_length_test, num_sentence_test = max_len(test)

    X_valid_sen = get_sentences(X_valid, max_length_valid, num_sentence_valid)
    X_test_sen = get_sentences(X_test, max_length_test, num_sentence_test)

    vocab_size = max(vocab_dict.values())
    tag_size = max(tag_dict.values())
    nwords = np.max([np.max(X_train), np.max(X_valid), np.max(X_test)])

    filename = 'data' + '.hdf5'
    with h5py.File(filename, "w") as f:
        f['X_train'] = X_train
        f['Y_train'] = Y_train
        f['X_valid'] = X_valid
        f['Y_valid'] = Y_valid
        f['X_test'] = X_test
        f['nwords'] = np.array([nwords])
        f['nclasses'] = np.array([tag_size])
        f['nfeatures'] = np.array([len(features)])
        f['X_valid_sen'] = X_valid_sen
        f['X_test_sen'] = X_test_sen

if __name__ == '__main__':
    sys.exit(main())
