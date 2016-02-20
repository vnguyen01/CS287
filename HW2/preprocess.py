
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

def gen_dictionary(data):
    vocab_list = []
    for data_set in data:
        vocab_list.extend([(word_process(word[2].lower())) for word in data_set if word != []])
    d = collections.defaultdict(list)
    counter = 2
    
    for word in vocab_list:
        if not d[word]:
            d[word] = counter
            counter = counter + 1
    d['PADDING'] = 0
    d['RARE'] = 1
    return d

def clean(data,tag_lookup, vocab):
    cleaned = [[vocab[word_process(word[2].lower())],1* str.isupper(word[2][0]), tag_lookup[word[3]]] if word != [] else '\n' for word in data]
    common = Counter(cleaned)    
    return cleaned, common

def clean_padding(data,tag_lookup, vocab, d_win, outputs):
    padded = []
    if outputs == True:
        padding = (np.ones((d_win/2, 3))*vocab['PADDING']).tolist()
    else:
        padding = (np.ones((d_win/2, 2))*vocab['PADDING']).tolist()
    padded.extend(padding)
    if outputs == True:
        for word in data:
            if word != []:
                padded.append([vocab[word_process(word[2].lower())],1*str.isupper(word[2][0]), tag_lookup[word[3]]])
            else:
                padded.extend(padding)
    else:
        for word in data:
            if word != []:
                padded.append([vocab[word_process(word[2].lower())],1*str.isupper(word[2][0])])
            else:
                padded.extend(padding) 
    return np.array(padded).astype(np.float)

def generate_windows(cleaned, vocab, d_win, padding, outputs):
    only_words = [word for word in cleaned if word[0] != 0]
    num_words = len(only_words)
    X_train = np.ones((num_words, d_win*2))*vocab[padding]
    Y_train = np.zeros(num_words)
    count = 0
    
    for i in range(0, len(cleaned)):
        if cleaned[i,0] != 0:
            X_train[count][:d_win] = map(int,cleaned[i-d_win/2:i+d_win/2+1, 0])
            X_train[count][d_win:] = map(int,cleaned[i-d_win/2:i+d_win/2+1, 1])
            if outputs == True:
                Y_train[count] = np.int(cleaned[i,2])
            count = count + 1
    if outputs == True:
        return X_train.astype(int), Y_train.astype(int)
    else:
        return X_train.astype(int)

FILE_PATHS = {"PTB": ("data/train.tags.txt",
                       "data/dev.tags.txt",
                       "data/test.tags.txt",
                       "data/tags.dict")}
args = {}

def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', help="Data set", type=str)
    parser.add_argument('window', help ="Window size", type=int)
    args = parser.parse_args(arguments)
    dataset = args.dataset
    window = args.window
    trainpath, validpath, testpath, tagspath = FILE_PATHS[dataset]

    if trainpath:
        train = list(csv.reader(open(trainpath, 'rb'), delimiter='\t'))
    if validpath:
        validate = list(csv.reader(open(validpath, 'rb'), delimiter='\t'))
    if testpath:
        test = list(csv.reader(open(testpath, 'rb'), delimiter='\t'))

    tags = dict(csv.reader(open(tagspath, 'rb'), delimiter='\t'))

    data = [train, validate, test]

    vocab = gen_dictionary(data)

    padded_train = clean_padding(data[0], tags, vocab, window, True)
    padded_validate = clean_padding(data[1], tags, vocab, window, True)
    padded_test = clean_padding(data[2], tags, vocab, window, False)

    train_input, train_output = generate_windows(padded_train, vocab, window, 'PADDING', True)
    valid_input, valid_output = generate_windows(padded_validate, vocab, window, 'PADDING', True)
    test_input = generate_windows(padded_test, vocab, window, 'PADDING', False)


    V = len(vocab) + 1
    print('Vocab size:', V, 'Window size:', window)

    C = np.max(train_output)

    filename = 'data' + '.hdf5'
    with h5py.File(filename, "w") as f:
        f['train_input'] = train_input
        f['train_output'] = train_output
        f['valid_input'] = valid_input
        f['valid_output'] = valid_output
        f['test_input'] = test_input
        f['nfeatures'] = np.array([V], dtype=np.int32)
        f['nclasses'] = np.array([C], dtype=np.int32)
        
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))