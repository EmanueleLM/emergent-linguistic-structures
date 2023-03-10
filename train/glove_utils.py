import numpy as np
from collections import defaultdict
import json
import math
import pickle
from numpy import linalg as LA
from pathlib import Path
import time

from embedding import Embedding

def pad_sequences(X, maxlen, emb_size=50):
    for i in range(len(X)):
        if len(X[i]) > maxlen:
            X[i] = X[i][:maxlen]
        elif len(X[i]) < maxlen:
            pad = np.zeros(shape=(maxlen-len(X[i]), emb_size))
            X[i] = np.append(X[i], pad, axis=0)
    return X


def index_to_word(word2index) :
    index2word = {value:key for key,value in word2index.items()}
    index2word[0] = '<PAD>'
    index2word[1] = '<START>'
    index2word[2] = '<UNK>'
    index2word[3] = '<UNUSED>'
    return index2word


# Based on https://github.com/guillaume-chevalier/GloVe-as-a-TensorFlow-Embedding-Layer/blob/master/GloVe-as-TensorFlow-Embedding-Tutorial.ipynb
def load_embedding(glove, try_first=''):
    '''
        Load word embeddings from file.
        try_first is the path to word_to_index_dict, index_to_word_dict, index_to_embedding_array files
        if they have already been pre-computed. Name should be try_first + {'w2v.dict', 'i2w.dict', 'i2e.npy'}
    '''
    w2v_file, i2w_file, i2e_file = Path(try_first+"w2v.dict"), Path(try_first+"i2w.dict"), Path(try_first+"i2e.npy") 
    if try_first != '':
        try:
            print(f"Trying to load pre-computed embeddings from `{try_first}`...")
            with open(str(w2v_file)) as f:
                word_to_index_dict = json.load(f)
                _UNK = 2
                word_to_index_dict = defaultdict(lambda: _UNK, word_to_index_dict)
            with open(str(i2w_file)) as f:
                index_to_word_dict = json.load(f)
            index_to_embedding_array = np.load(str(i2e_file), allow_pickle=True)
        except:
            print(f"Failed to load pre-computed embeddings, now creating and storing the embeddings...")
            word_to_index_dict = dict()
            index_to_embedding_array = []        
            with open(glove, 'r', encoding="utf-8") as glove_file:
                for (i, line) in enumerate(glove_file):
                    split = line.split(' ')            
                    word = split[0]            
                    representation = split[1:]
                    representation = np.array(
                        [float(val) for val in representation]
                    )
                    # use +4 because actual word indexes start at 4 while indexes 0,1,2,3 are for
                    # <PAD>, <START>, <UNK> and <UNUSED>
                    word_to_index_dict[word] = i+4
                    index_to_embedding_array.append(representation)
            _WORD_NOT_FOUND = [0.0]* len(representation)  # Empty representation for unknown words.
            _PAD = 0
            _START = 1
            _UNK = 2
            word_to_index_dict['<PAD>'] = 0
            word_to_index_dict['<START>'] = 1
            word_to_index_dict['<UNK>'] = 2
            word_to_index_dict['<UNUSED>'] = 3
            word_to_index_dict['<pad>'] = 4
            word_to_index_dict['<start>'] = 5
            word_to_index_dict['<unk>'] = 6
            word_to_index_dict['<unused>'] = 7
            word_to_index_dict = defaultdict(lambda: _UNK, word_to_index_dict)
            index_to_word_dict = index_to_word(word_to_index_dict)
            # three 0 vectors for <PAD>, <START> and <UNK>
            index_to_embedding_array = np.array(4*[_WORD_NOT_FOUND] + index_to_embedding_array )
            # Save dictionaries to fasten the procedure
            if not(w2v_file.is_file() and i2w_file.is_file() and i2e_file.is_file()):
                print(f"Storing pre-computed embeddings to {try_first}")
                with open(str(w2v_file), 'w') as f:
                    json.dump(word_to_index_dict, f)
                with open(str(i2w_file), 'w') as f:
                    json.dump(index_to_word_dict, f)
                np.save(str(i2e_file), index_to_embedding_array)
    else:
        word_to_index_dict = dict()
        index_to_embedding_array = []        
        with open(glove, 'r', encoding="utf-8") as glove_file:
            for (i, line) in enumerate(glove_file):
                split = line.split(' ')            
                word = split[0]            
                representation = split[1:]
                representation = np.array(
                    [float(val) for val in representation]
                )
                # use +4 because actual word indexes start at 4 while indexes 0,1,2,3 are for
                # <PAD>, <START>, <UNK> and <UNUSED>
                word_to_index_dict[word] = i+4
                index_to_embedding_array.append(representation)
        _WORD_NOT_FOUND = [0.0]* len(representation)  # Empty representation for unknown words.
        _PAD = 0
        _START = 1
        _UNK = 2
        word_to_index_dict['<PAD>'] = 0
        word_to_index_dict['<START>'] = 1
        word_to_index_dict['<UNK>'] = 2
        word_to_index_dict['<UNUSED>'] = 3
        word_to_index_dict['<pad>'] = 4
        word_to_index_dict['<start>'] = 5
        word_to_index_dict['<unk>'] = 6
        word_to_index_dict['<unused>'] = 7
        word_to_index_dict = defaultdict(lambda: _UNK, word_to_index_dict)
        index_to_word_dict = index_to_word(word_to_index_dict)
        # three 0 vectors for <PAD>, <START> and <UNK>
        index_to_embedding_array = np.array(4*[_WORD_NOT_FOUND] + index_to_embedding_array )
    return word_to_index_dict, index_to_word_dict, index_to_embedding_array


def load_binary_embedding(words):
    num_digits = math.ceil(math.log2((1+len(words))))
    word_to_index_dict = dict()
    index_to_embedding_array = []
    for i,w in zip(range(len(words)), words):
        representation = [float(digit) for digit in bin(i+1)[2:]][::-1]
        representation.extend([0. for _ in range(num_digits-len(representation))])
        index_to_embedding_array.append(representation[::-1])
        word_to_index_dict[w] = i+4
    _WORD_NOT_FOUND = [0.0]* num_digits  # Empty representation for unknown words.
    _PAD = 0
    _START = 1
    _UNK = 2
    word_to_index_dict['<PAD>'] = 0
    word_to_index_dict['<START>'] = 1
    word_to_index_dict['<UNK>'] = 2
    word_to_index_dict['<UNUSED>'] = 3
    word_to_index_dict = defaultdict(lambda: _UNK, word_to_index_dict)
    index_to_word_dict = index_to_word(word_to_index_dict)
    index_to_embedding_array = np.array(4*[_WORD_NOT_FOUND] + index_to_embedding_array )
    return word_to_index_dict, index_to_word_dict, index_to_embedding_array


def load_syn_dict(filename = 'data/syn_dict/syn_dict_glove300.pickle', N = 10):
    '''
        Load cached synonyms dictionary.
    '''
    try:
        file = open(filename, 'rb')
        syn_dict = pickle.load(file)
        syn_dict = {word: neighbors[:N] for word, neighbors in syn_dict.items()}
        return syn_dict
    except:
        print("ERROR: Could not load synonyms dictionary.")
    return dict()


def load_dist_dict(filename = 'data/syn_dict/dist_dict_glove300.pickle', N = 10):
    '''
        Load cached dictionary with distances to nearest neighbors.
    '''
    try :
        file = open(filename, 'rb')
        dist_dict = pickle.load(file)
        dist_dict = {word: distances[:N] for word, distances in dist_dict.items()}
        return dist_dict
    except:
        print("ERROR: Could not load distances to nearest neighbors dictionary")
    return dict()
