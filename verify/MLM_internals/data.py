import matplotlib as plt
import networkx as nx
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import seaborn as sns
import sys
import tqdm

from collections import deque, defaultdict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from pandas import read_csv
from sklearn.decomposition import KernelPCA, PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize as scipy_normalize
from tensorflow.keras.datasets import imdb
from transformers import AdamW, BertTokenizerFast, BertForMaskedLM, BertConfig
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig

sys.path.append('./../'); sys.path.append('./../../')
from linguistic_augmentation import shallow_negation, mixed_sentiment, sarcasm

def load_SST(num_samples=-1, return_text=False):
    # Load STT dataset (eliminate punctuation, add padding etc.)
    X_train = read_csv('./../../data/datasets/SST_2/training/SST_2__FULL.csv', sep=',',header=None).values
    X_test = read_csv('./../../data/datasets/SST_2/eval/SST_2__TEST.csv', sep=',',header=None).values
    y_train, y_test = [], []
    for i in range(len(X_train)):
        r, s = X_train[i]  # review, score (comma separated in the original file)
        X_train[i][0] = [w.lower() for w in r.translate(str.maketrans('', '', string.punctuation)).strip().split(' ')]
        y_train.append((0 if s.strip()=='negative' else 1))
    for i in range(len(X_test)):
        r, s = X_test[i]
        X_test[i][0] = [w.lower() for w in r.translate(str.maketrans('', '', string.punctuation)).strip().split(' ')]
        y_test.append((0 if s.strip()=='negative' else 1))
    X_train, X_test = X_train[:,0], X_test[:,0]
    n = num_samples  # you may want to take just some samples (-1 to take them all)
    X_train = X_train[:n]
    X_test = X_test[:n]
    y_train = y_train[:n]
    y_test = y_test[:n]
    if return_text is False:
        raise(NotImplementedError("Dataset can't be loaded in this way and must be splitted, check train_STT.py for an example"))
    else:
        return (X_train, y_train), (X_test, y_test)

def load_imdb():
    train,test = imdb.load_data(num_words=10000, index_from=3)
    X_train, y_train = train
    X_test, y_test = test
    return (X_train, y_train), (X_test, y_test)

def get_texts(dataset, max_texts, cap_length=50):
    """
    max_texts is the number of texts to be returned
    cap_length affects only imdb and wikipedia and caps the length of each input
    """
    print(f"Start collecting paraphrases from `{dataset}` dataset")
    paraphrases = []
    if dataset == 'ppdb':
        # get some para-phrases
        database = './../../data/databases/ppdb-2.0-s-phrasal'
        num_lines = sum(1 for line in open(database))
        if random_sampling is False:
            with open(database, 'r+') as file_:
                while len(paraphrases)!=max_texts:
                    src = file_.readline().split('|||')
                    phrase = src[1].strip().split()
                    paraphrases += [' '.join(phrase)]
        else:
            for _ in tqdm.tqdm(range(max_texts)):
                rn = np.random.randint(0, num_lines)
                src = linecache.getline(database, rn).split('|||')
                phrase = src[1].strip().split()
                paraphrases += [' '.join(phrase)]
    elif dataset == 'sst':
        (input_texts, input_labels), (_, _) = load_SST(return_text=True)
        num_lines = len(input_texts)
        n=0
        while len(paraphrases)!=max_texts:
            if random_sampling is True:
                n = np.random.randint(0, num_lines)
            else:
                n += 1
            src = input_texts[n]
            paraphrases += [' '.join(src)]   
    elif dataset == 'imdb':
        (X_train, y_train), (_, _) = load_imdb()
        word_to_id = imdb.get_word_index()
        word_to_id = {k:(v+3) for k,v in word_to_id.items()}
        word_to_id["<PAD>"] = 0; word_to_id["<START>"] = 1; word_to_id["<UNK>"] = 2; word_to_id["<UNUSED>"] = 3
        id_to_word = {value:key for key,value in word_to_id.items()}
        num_lines = len(X_train)
        n = 0
        while len(paraphrases)!=max_texts:
            if random_sampling is True:
                n = np.random.randint(0, num_lines)
            else:
                n += 1
            src = [id_to_word[id_] for id_ in X_train[n][1:cap_length]]  # remove <START> token
            paraphrases += [' '.join(src)]
    elif dataset == 'wikipedia':
        database = './../../data/datasets/wikipedia/wikipedia.txt'
        num_lines = sum(1 for line in open(database))
        if random_sampling is False:
            with open(database, 'r+') as file_:
                while len(paraphrases)!=max_texts:
                    src = file_.readline().split(' ')[:cap_length]
                    paraphrases += [' '.join(src)]
        else:
            for _ in tqdm.tqdm(range(max_texts)):
                rn = np.random.randint(0, num_lines)
                src = linecache.getline(database, rn).split(' ')[:cap_length]
                paraphrases += [' '.join(src)]
    elif 'tkn' in dataset:  # dataset from `the king is naked`
        if dataset == 'tkn-negation':
            X,_ = shallow_negation()
        elif dataset == 'tkn-mixed':
            X,_ = mixed_sentiment()
        elif dataset == 'tkn-sarcasm':
            X,_ = sarcasm()
        else:
            raise Exception(f"{dataset} is not a valid value for `dataset`.")
        if random_sampling is True:
            np.random.shuffle(X)
        for x in X[:max_texts]:
            paraphrases += [' '.join(x)]
    else:
        raise Exception(f"{dataset} is not a valid value for `dataset`.")
    
    print(f"{len(paraphrases)} texts have been collected")
    return paraphrases

def floyd(G, nV):
    """
    https://favtutor.com/blogs/floyd-warshall-algorithm
    """
    dist = list(map(lambda p: list(map(lambda q: q, p)), G))
    # Adding vertices individually
    for r in range(nV):
        for p in range(nV):
            for q in range(nV):
                if p==q:
                    dist[p][q] = 0
                else:
                    dist[p][q] = min(dist[p][q], dist[p][r] + dist[r][q])
    return np.array(dist)

def conll2matrix(tab, separator, head_position, maxlen, distance=True, vinf=999):
    """
    Turn a table of conll-compliant dataset into a matrix of relative distances.
    Use the tree to build a distance matrix where each word has a 1 for an ingoing/outgoing edge.
    Use 0 for the diagonal and to pad shorter sentences (at some point you might
    think to use sparse matrices as representations).
    Return the input string and the `distance` matrix if distance=True, otherwise the adjacency matrix.

    separator is the separator used in the conll file to separate columns, 
    head_position is the position of the `HEAD` value,
    maxlen is the maximum length of a sequence,
    distance is a boolean to return the adjacency/distance when False/True
    vinf is the value used to denote no-connection (ignored if distance is False),
    """
    s = []
    M = np.zeros((maxlen, maxlen)) + (vinf if distance is True else 0)
    #print(tab)
    for i,row in enumerate(tab):
        #print(row)
        if i==maxlen:
            break
        x = row.split(separator)
        #print(x)
        try:
            idx = int(x[0])-1
        except:
            continue
        head = int(x[head_position])-1
        #print(x[1])
        s += [str(x[1])]
        if 0 < head < maxlen:
            M[idx][head] = M[head][idx] = 1
    if distance is True:
        M = floyd(M, maxlen)
    #print(s, M)
    return ' '.join(s), M

def conll2root(tab, separator, head_position, maxlen):
    """
    Return from a table of conll-compliant dataset the index of the root, or the minimum element, 
     if root is not present.

    separator is the separator used in the conll file to separate columns, 
    head_position is the position of the `HEAD` value,
    maxlen is the maximum length of a sequence,
    """
    s = []
    R = np.zeros((maxlen))
    root = -1
    new_min = maxlen-1
    #print(tab)
    for i,row in enumerate(tab):
        #print(row)
        if i==maxlen:
            break
        x = row.split(separator)
        #print(x)
        s += [str(x[1])]
        #print(x[head_position])
        try:
            int(x[head_position])
        except:
            continue
        if int(x[head_position]) < new_min:
            root = i
            new_min = int(x[head_position])
        #print(root, new_min)
    assert root != -1
    R[root] = 1
    #print(s, R)
    return ' '.join(s), R

def get_pos_from_conlldatasets(dataset, nsamples=-1, filter_ratio=0.5):
    """
    Collect a conll2009 compliant list of datasets, and extract the syntax tree through the 
    `HEAD (index of syntactic parent, 0 for ROOT)` column. 
    Return the input strings as list of n strings, a list of list of pos-tag sentences (where each POS correspond to a number),
     a dictionary of POS:frequency and a index2POS dictionary.
    """
    if isinstance(dataset, str):
        datasets = [dataset]
    else:
        assert isinstance(dataset, list)
        datasets = list(dataset)
        print(datasets)

    POS_FREQ = {}  # pos:frequency
    S = []  # list of sentences
    S_POS = []  # each word, its pos-tag
    n = 0
    for dataset in datasets:
        prefix_data = './../../../data/datasets/conll/'
        if dataset == 'ted':
            pos_position = 3
            prefix_data += 'ted/dataset.dep'
        elif dataset == 'ud-english-pud':
            pos_position = 4
            prefix_data += 'ud-english-pud/dataset.dep'
        elif dataset == 'ud-english-lines':
            pos_position = 4
            prefix_data += 'ud-english-lines/dataset.dep'
            print(f"[WARNING]: {dataset} dataset has no proper POS tags, be careful to judge the results in the correct way.")
        elif dataset == 'en-universal':
            pos_position = 4
            prefix_data += 'en-universal/dataset.dep'
        elif dataset == 'ud-english-gum':
            pos_position = 4
            prefix_data += 'ud-english-gum/dataset.dep'
        elif dataset == 'ud-english-ewt':
            pos_position = 4
            prefix_data += 'ud-english-ewt/dataset.dep'
        else:
            raise Exception(f"{dataset} is not a valid conll dataset.")
        
        with open(prefix_data, 'r+') as file_:
            file_length = num_lines = sum(1 for line in open(prefix_data))
            s, s_pos = [], []
            for i,line in tqdm.tqdm(enumerate(file_)):
                if n == nsamples:
                    break
                if line != '\n':
                    x = line.split('\t')
                    p = str(x[pos_position])
                    if p not in POS_FREQ.keys():
                        POS_FREQ[p] = 1
                    else:
                        POS_FREQ[p] += 1
                    s += [str(x[1])]
                    s_pos += [p]
                else:
                    n += 1
                    S += [' '.join(s)]; s = []
                    S_POS += [s_pos]; s_pos = []
                #print(str(x[1]), ' - ', p)
 

    POS_FREQ = sorted(POS_FREQ.items(), key=lambda kv: kv[1], reverse=True)
    # filter unfrequent POS-tags
    POS_FREQ = POS_FREQ[:int(len(POS_FREQ)*filter_ratio)]
    pos2index = {v[0]:1+p for p,v in enumerate(POS_FREQ)}
    pos2index = defaultdict(lambda: 0, pos2index)  # default value for filtered data is 0
    S_TAGGED = []
    for s,pos in zip(S, S_POS):
        s_tagged = []
        for _,tag in zip(s, pos):
            s_tagged += [pos2index[tag]]
        S_TAGGED += [s_tagged]
    #print(pos)
    return S, S_POS, POS_FREQ, S_TAGGED


def get_trees_from_conlldatasets(dataset, maxlen=25, nsamples=-1):
    """
    Collect a conll2009 compliant list of datasets, and extract the syntax tree through the 
    `HEAD (index of syntactic parent, 0 for ROOT)` column. 
    Return the input strings and the `distance` matrix.
    """
    if isinstance(dataset, str):
        datasets = [dataset]
    else:
        assert isinstance(dataset, list)
        datasets = list(dataset)
        print(datasets)

    M = np.zeros((1, maxlen, maxlen))  # contains the distance matrices
    S = []  # contains the input texts
    for dataset in datasets:
        prefix_data = './../../../data/datasets/conll/'
        if dataset == 'ted':
            prefix_data += 'ted/dataset.dep'
        elif dataset == 'ud-english-pud':
            prefix_data += 'ud-english-pud/dataset.dep'
        elif dataset == 'ud-english-lines':
            prefix_data += 'ud-english-lines/dataset.dep'
        elif dataset == 'en-universal':
            prefix_data += 'en-universal/dataset.dep'
        elif dataset == 'ud-english-ewt':
            prefix_data += 'ud-english-ewt/dataset.dep'
        elif dataset == 'ud-english-gum':
            prefix_data += 'ud-english-gum/dataset.dep'
        else:
            raise Exception(f"{dataset} is not a valid conll dataset.")

        # Collect the data
        separator = '\t'
        with open(prefix_data, 'r+') as file_:
            file_length = num_lines = sum(1 for line in open(prefix_data))    
            tab = []
            for i,line in tqdm.tqdm(enumerate(file_)):
                if nsamples != -1:
                    if len(M) > nsamples+1:
                        break
                if line != '\n':
                    tab += [line]
                else:
                    s, m = conll2matrix(tab, separator, head_position=6, maxlen=maxlen)
                    #print(s,m)
                    #print(m.shape)
                    M = np.append(M, m.reshape(1,*m.shape), axis=0)
                    S += [s]
                    tab = []    
    return S, M[1:]  # discard the first input (it's a numpy token to prevent np.append from failing)      

def get_roots_from_conlldatasets(dataset, maxlen=25, nsamples=-1):
    """
    Collect a conll2009 compliant list of datasets, and extract the position of the root through the 
    `HEAD (index of syntactic parent, 0 for ROOT)` column. 
    Return the input strings and the position of the root: in case the root is not between 0 and maxlen, return the index of the element closest to the root.
    """
    if isinstance(dataset, str):
        datasets = [dataset]
    else:
        assert isinstance(dataset, list)
        datasets = list(dataset)
        print(datasets)

    R = np.zeros((1, maxlen))  # contains the distance matrices
    S = []  # contains the input texts
    for dataset in datasets:
        prefix_data = './../../../data/datasets/conll/'
        if dataset == 'ted':
            prefix_data += 'ted/dataset.dep'
        elif dataset == 'ud-english-pud':
            prefix_data += 'ud-english-pud/dataset.dep'
        elif dataset == 'ud-english-lines':
            prefix_data += 'ud-english-lines/dataset.dep'
        elif dataset == 'en-universal':
            prefix_data += 'en-universal/dataset.dep'
        elif dataset == 'ud-english-ewt':
            prefix_data += 'ud-english-ewt/dataset.dep'
        elif dataset == 'ud-english-gum':
            prefix_data += 'ud-english-gum/dataset.dep'
        else:
            raise Exception(f"{dataset} is not a valid conll dataset.")

        # Collect the data
        separator = '\t'
        with open(prefix_data, 'r+') as file_:
            file_length = num_lines = sum(1 for line in open(prefix_data))    
            tab = []
            for i,line in tqdm.tqdm(enumerate(file_)):
                if nsamples != -1:
                    if len(M) > nsamples+1:
                        break
                if line != '\n':
                    tab += [line]
                else:
                    s, r = conll2root(tab, separator, head_position=6, maxlen=maxlen)
                    #print(s,m)
                    #print(m.shape)
                    R = np.append(R, r.reshape(1, len(r)), axis=0)
                    S += [s]
                    tab = []    
    return S, R[1:]  # discard the first input (it's a numpy token to prevent np.append from failing)      


def get_depths_from_conlldatasets(dataset, maxlen=25, nsamples=-1):
    """
    Collect a conll2009 compliant list of datasets, and extract the depth of the root (up to the maxlen samples) through the 
    `HEAD (index of syntactic parent, 0 for ROOT)` column. 
    Return the input strings and the depth (one-hot encoding) of the tree (up to the maxlen samples).
    Please notice that this function mixes conll2matrix and conll2root functions, the building blocks of get_trees_from_conlldatasets 
    and get_roots_from_conlldatasets functions.
    """
    if isinstance(dataset, str):
        datasets = [dataset]
    else:
        assert isinstance(dataset, list)
        datasets = list(dataset)
        print(datasets)

    D = np.zeros((1, maxlen))  # contains the distance matrices
    S = []  # contains the input texts
    for dataset in datasets:
        prefix_data = './../../../data/datasets/conll/'
        if dataset == 'ted':
            prefix_data += 'ted/dataset.dep'
        elif dataset == 'ud-english-pud':
            prefix_data += 'ud-english-pud/dataset.dep'
        elif dataset == 'ud-english-lines':
            prefix_data += 'ud-english-lines/dataset.dep'
        elif dataset == 'en-universal':
            prefix_data += 'en-universal/dataset.dep'
        elif dataset == 'ud-english-ewt':
            prefix_data += 'ud-english-ewt/dataset.dep'
        elif dataset == 'ud-english-gum':
            prefix_data += 'ud-english-gum/dataset.dep'
        else:
            raise Exception(f"{dataset} is not a valid conll dataset.")

        # Collect the data
        local_vinf = 999
        separator = '\t'
        with open(prefix_data, 'r+') as file_:
            file_length = num_lines = sum(1 for line in open(prefix_data))    
            tab = []
            for i,line in tqdm.tqdm(enumerate(file_)):
                if nsamples != -1:
                    if len(M) > nsamples+1:
                        break
                if line != '\n':
                    tab += [line]
                else:
                    s, m = conll2matrix(tab, separator, head_position=6, maxlen=maxlen, vinf=local_vinf)
                    _, r = conll2root(tab, separator, head_position=6, maxlen=maxlen)
                    try:
                        m_numpy = np.array(m)
                        m_numpy[m_numpy==local_vinf] = -1
                        d_idx = int(max(m_numpy[np.argmax(r)]))
                    except:  # root is father than the maxlen-th element
                        d_idx = 0
                    d = np.zeros((1, maxlen))
                    d[0,d_idx] = 1
                    #print(s,m)
                    #print(m.shape)
                    D = np.append(D, d, axis=0)
                    S += [s]
                    tab = []    
    return S, D[1:]  # discard the first input (it's a numpy token to prevent np.append from failing)    

"""
def get_entities_from_conlldatasets(dataset, maxlen=25, nsamples=-1, filter_ratio=0.5):
    if isinstance(dataset, str):
        datasets = [dataset]
    else:
        assert isinstance(dataset, list)
        datasets = list(dataset)
        print(datasets)

    POS = np.zeros((1, maxlen))  # contains the POS vectors
    POD_dictionary = {}
    S = []  # contains the input texts
    for dataset in datasets:

        # Collect the data
        pos = get_pos_from_conlldatasets(dataset, maxlen, nsamples)
        pos 
        for k,v in pos.items():
            if k not in POS.keys():
                POS[k] = 1
            else:
                POS[k] += v

    POS = sorted(POS.items(), key=lambda kv: kv[1], reverse=True)
    POS = POS[:filter_ratio]
    
    return S, M[1:]  # discard the first input (it's a numpy token to prevent np.append from failing)
"""

def draw_graph(words, M, savename=''):
    """
    Draw the syntax tree as a graph. Takes as input a list of words (for labelling)
    and the matrix of the distances of each word-pair M.
    Outputs the graph, which can be saved on file.
    """
    plt.cla()
    if M.ndim == 3:
        M = M[0,:,:]
    assert len(words) <= M.shape[0] <= M.shape[1]

    # Deal with repeated words
    for w in words:
        cnt = words.count(w)
        if cnt > 1:
            i = 0
            while i != cnt:
                for j,w_nm in enumerate(words):
                    if w == w_nm:
                        words[j] += f'_({i})'
                        i += 1
    G = nx.Graph()
    for i,w in enumerate(words):
        for j,w in enumerate(words):
            if M[i,j] == 1:
                G.add_edge(words[i], words[j], weight=M[i,j])

    # Draw edges
    edges = [(u, v) for (u, v, d) in G.edges(data=True)]

    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=100, node_size_color='blue', alpha=0.1)
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=1, edge_color='red', alpha=1.)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif', font_color='black', alpha=1.)
    plt.axis('off')

    if len(savename) != '':
        print(f"Saving file to {savename}")
        plt.savefig(savename)
    else:
        plt.show()
    plt.cla()
