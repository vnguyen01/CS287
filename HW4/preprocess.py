#!/usr/bin/env python

"""Part-Of-Speech Preprocessing

HDF5 tags:

-train_rnn_X: see pset spec for RNN preprocessing
-train_rnn_Y: see pset spec for RNN preprocessing
-train_X_sequence: the original train sequence indices all in one array
-train_Y_sequence: same thing where the output is whether or not the next character is space (1: non-space, 2:space)
-valid_reduced_X: a smaller version of the actual validation sequence but in the same format as train_X_sequence so that you can see how your training compares to validation
-valid_reduced_Y same as train_Y_sequence except for the reduced validation set
-windows_train: windowed inputs (originally 5) with padding (called fluff in this code) at the beggining for the purposes of the count based and NNLM
-windows_valid: same as above but for reduced valid sequence
-test: test input, a 2D array. Since sequences are of variable length, the array is dimension nxd where d is the length of the longest sequence. the rest of the sequences are padded with the stop symbol '</s>'
-valid_kaggle_with_spaces: 2D array in the same format as test with stop symbol padding, has spaces
-valid_kaggle_without_spaces: same as above except all spaces have been removed (for predicition purposes)
-valid_answers: 1d array (list) of counts of number of spaces

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


FILE_PATHS = {"full": ("data/train_chars.txt",
                      "data/valid_chars.txt",
                      "data/valid_chars_kaggle_answer.txt",
                      "data/valid_chars_kaggle.txt",
                      "data/test_chars.txt",)}

def main(arguments):
    global args
    parser = argparse.ArgumentParser(description=__doc__, 
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('backprop_l', help="Backprop Length", type=int)
    parser.add_argument('batch_size', help="Batch Size", type=int)
    parser.add_argument('win_size', help="Window Size", type=int)
    args = parser.parse_args(arguments)

    l = args.backprop_l
    b = args.batch_size
    w_size = args.win_size

    train_path, valid_path, valid_kaggle_answer_path, valid_kaggle_path, test_path = FILE_PATHS["full"]

    train_X = (list(csv.reader(open(train_path, 'rb'), delimiter=' ')))[0]
    valid_reduced_X = list(csv.reader(open(valid_path, 'rb'), delimiter=' '))[0]
    valid_kaggle  = list(csv.reader(open(valid_kaggle_path, 'rb'), delimiter=' '))
    valid_answers = (list(csv.reader(open(valid_kaggle_answer_path, 'rb'), delimiter=',')))[1:]
    test = list(csv.reader(open(test_path, 'rb'), delimiter=' '))

    char_dict = dict.fromkeys(string.ascii_lowercase, 0)

    def gen_windows(xdata, window_size):
        windows = np.zeros((len(xdata), window_size))
        fluffed = ([char_dict['</s>']] * (window_size-1)) + xdata
        for i in range(0,len(fluffed)-window_size+1):
            window = fluffed[i:i+window_size]
            if window != [char_dict['</s>']]*window_size:
                windows[i]= window
        return windows

    def get_outputs(X):
        #space = 2, non-space = 1
        outputs = np.zeros(len(X))
        for i in range(0, len(X)-1):
            if X[i+1] == char_dict['<space>']:
                outputs[i] = 2
            else:
                outputs[i] = 1
        outputs[-1] = 1
        return outputs
    
    count = 1
    for key in char_dict.keys():
        char_dict[key] = count
        count = count + 1
    char_dict['<s>'] = count
    count = count + 1

    for i in range(0,len(train_X)):
        char = train_X[i]
        if char not in char_dict.keys():
            char_dict[char] = count
            count = count + 1
        train_X[i] = char_dict[train_X[i]]
        
    for i in range(0,len(valid_reduced_X)):
        char = valid_reduced_X[i]
        if char not in char_dict.keys():
            char_dict[char] = count
            count = count + 1
        valid_reduced_X[i] = char_dict[valid_reduced_X[i]]
        
    for i in range(0,len(valid_kaggle)):
        for j in range(0,len(valid_kaggle[i])):
            char = valid_kaggle[i][j]
            if char not in char_dict.keys():
                char_dict[char] = count
                count = count + 1
            valid_kaggle[i][j] = char_dict[valid_kaggle[i][j]]
            
    for i in range(0,len(test)):
        for j in range(0,len(test[i])):
            char = test[i][j]
            if char not in char_dict.keys():
                char_dict[char] = count
                count = count + 1
            test[i][j] = char_dict[test[i][j]]

    n = len(train_X)
    padding = np.ones((l*b - (n % (l*b)))%(l*b))
    padded_train_X = np.append(train_X,padding)
    padded_train_Y = get_outputs(padded_train_X)
    padded_n = len(padded_train_X)
    train_rnn_X = np.zeros((padded_n/(b*l), b, l))
    train_rnn_Y = np.zeros((padded_n/(b*l), b, l))
    for j in range(0,len(train_rnn_X[0])):
        for i in range(0,len(train_rnn_X)):
            segment_X = padded_train_X[l*len(train_rnn_X)*j + l*i:l*len(train_rnn_X)*j + l*i + l]
            segment_Y = padded_train_Y[l*len(train_rnn_X)*j + l*i:l*len(train_rnn_X)*j + l*i + l]
            train_rnn_X[i][j] = segment_X
            train_rnn_Y[i][j] = segment_Y
            
    valid_reduced_Y = get_outputs(valid_reduced_X)
    train_Y = get_outputs(train_X)

    vk = copy.deepcopy(valid_kaggle)
    for i in range(0,len(vk)):
        vk[i] = [x for x in vk[i] if x != 28]
        
    length = len(sorted(test,key=len, reverse=True)[0])
    length = max(len(sorted(valid_kaggle ,key=len, reverse=True)[0]), length)
    test = np.array([xi+[char_dict['</s>']]*(length-len(xi)) for xi in test])
    valid_kaggle_with_spaces = np.array([xi+[char_dict['</s>']]*(length-len(xi)) for xi in valid_kaggle])
    valid_kaggle_without_spaces = np.array([xi+[char_dict['</s>']]*(length-len(xi)) for xi in vk])


    windows_train = gen_windows(train_X, w_size)
    windows_valid = gen_windows(valid_reduced_X, w_size)

    filename = 'data' + '.hdf5'
    with h5py.File(filename, "w") as f:
        f['train_rnn_X'] = train_rnn_X
        f['train_rnn_Y'] = train_rnn_Y
        f['train_X_sequence'] = np.array(train_X)
        f['train_Y'] = train_Y
        f['valid_reduced_X'] = np.array(valid_reduced_X)
        f['valid_reduced_Y'] = valid_reduced_Y
        f['windows_train'] = windows_train
        f['windows_valid'] = windows_valid
        f['test'] = test
        f['valid_kaggle_with_spaces'] = valid_kaggle_with_spaces
        f['valid_kaggle_without_spaces'] = valid_kaggle_without_spaces
        f['valid_answers'] = (np.array(valid_answers).astype(int))[:,1]
        f['nfeatures'] = np.array([len(char_dict) + 1])
        f['nclasses'] = np.array([2])

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
