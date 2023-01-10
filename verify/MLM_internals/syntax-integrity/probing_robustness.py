"""
RQ. Are representations robust to syntax manioulations? In other words, are NLP models encoding robust representations?
P1-task: Structural probe. Given a sentence, reconstruct the syntax tree via a representation' embedding (a conll dataset is the ground truth).
T1-task: POS-tagging. Given a sentence, identify the POS tags via a representation' embedding (a conll dataset is the ground truth).
R1-task: Root identification. Given a sentence, identify the root of the tree via a representation' embedding (a conll dataset is the ground truth).

For each task, a classification/regression neural network is attached to a Large Language Model (could be also referred in the code as llm/MLM, i.e., Masked Language Model),
that learns to map X->Y, where X is the embedded representation of an input sentence and Y is the target label. It is the syntax tree (or the distance matrix) for P1, 
a multi-class classification vector for POS-tag (T1) and root identification (R1). 
A few metrics are used to judge each model on the tasks, like Spearman correlation (P1), accuracy (T1, R1). Please refer to the logs of a running file.
"""
import argparse
import copy as cp
import itertools
import linecache
import numpy as np
import os
import random
import re
import string
import tensorflow as tf
import torch
import torch.nn as nn
import seaborn as sns
import sys
import tqdm

import matplotlib.pyplot as plt
from keras.utils import to_categorical
from pathlib import Path
from scipy import spatial
from sklearn.preprocessing import normalize as scipy_normalize
from tensorflow.keras.activations import softmax
from tensorflow.keras.layers import Dense, Flatten, Input, Conv1D, Conv2D, Bidirectional, LSTM, concatenate, Reshape, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.random import set_seed
from transformers import AdamW, BertTokenizer, BertModel, BertConfig, BertTokenizerFast, BertForMaskedLM
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig

from generate_perturbations import sequential_interventions, wordnet_perturbations
from p1_metrics import UUAS, same_distance, spearman_pairwise, spearman
from r1_metrics import accuracy as r1_accuracy
from t1_metrics import accuracy as t1_accuracy
from d1_metrics import same_distance as d1_same_distance
from d1_metrics import spearman as d1_spearman
sys.path.append('./../')
from data import get_trees_from_conlldatasets, get_pos_from_conlldatasets, get_roots_from_conlldatasets, get_depths_from_conlldatasets
sys.path.append('./../../../train')
from glove_utils import load_embedding
from BERTModels import SentimentalBERT

def probing_loss(H, M, B, max_length, embedding_dim, k):
    """
    Probing loss, eq. (5) of the paper "Emergent linguistic structure in artificial neural networks trained by self-supervision", PNAS|December 1, 2020|vol. 117|no. 48.
    The normalizing factor 1/|s^l|**2 is in our case a constant 1/max_length**2.
    input1: H is the matrix of the inputs, one vector for each word, shape (max_length, embedding_dim).
    input2: M is the distance-tree matrix, (shape max_length**2).
    output: B is the output of the n.n., k,shape embedding_dim.
    """
    # Reshape inputs before feeding the graph
    H = tf.reshape(H, (-1, max_length, embedding_dim))
    M = tf.reshape(M, (-1, max_length, max_length))
    B = tf.reshape(B, (k, embedding_dim))
    op1 = tf.expand_dims(-H, axis=1)
    op2 = tf.expand_dims(H, axis=2)
    H_diffs = op1 + op2
    # Compute B(h_i-h_j) \forall(i,j)
    B_H_dot = tf.tensordot(H_diffs, tf.transpose(B), axes=1)
    # Compute the L-2 norm of B(h_i-h_j) along last axis (dim k)
    B_H_l2 = tf.norm(B_H_dot, ord=2, axis=-1)
    # Compute the difference between each pair of words in the tree and B(h_i-h_j)
    L_abs = tf.abs(M-B_H_l2)/max_length**2  # normalization constant
    Loss = tf.math.reduce_sum(L_abs)
    return Loss


def p1_custom_softmax(t):
    global n_pos
    import tensorflow.keras as K
    n_classes = n_pos  # the diameter of the line-graph tree
    sh = t.shape
    partial_sm = []
    #print(sh[1] // n_classes)
    for i in range(sh[1] // n_classes):
        partial_sm.append(softmax(t[:, i*n_classes:(i+1)*n_classes]))
    return concatenate(partial_sm)

def t1_custom_softmax(t):
    global POS
    import tensorflow.keras as K
    n_classes = int(np.max(POS)+1)
    sh = t.shape
    partial_sm = []
    for i in range(sh[1] // n_classes):
        partial_sm.append(softmax(t[:, i*n_classes:(i+1)*n_classes]))
    return concatenate(partial_sm)

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


# Parse command line args
parser = argparse.ArgumentParser()
parser.add_argument("-a", "--architecture", dest="architecture", type=str, default='fc',
                    help="Architecture of the model that learns the H->C mapping (`fc`, `cnn`, `rnn`, `cnn2d`).")
parser.add_argument("-actp1", "--activationp1", dest="activation_p1", type=str, default='linear',
                    help="Activation of each network layer for the structural probe task (`linear`, `tanh`, `relu`, etc.).")
parser.add_argument("-actt1", "--activationt1", dest="activation_t1", type=str, default='linear',
                    help="Activation of each network layer for the POS-tag task (`linear`, `tanh`, `relu`, etc.).")
parser.add_argument("-actr1", "--activationr1", dest="activation_r1", type=str, default='linear',
                    help="Activation of each network layer for the Root classification task (`linear`, `tanh`, `relu`, etc.).")
parser.add_argument("-actd1", "--activationd1", dest="activation_d1", type=str, default='linear',
                    help="Activation of each network layer for the Tree-depth classification task (`linear`, `tanh`, `relu`, etc.).")
parser.add_argument("-l", "--layer-llm", dest="layer_llm", type=int, default=-9,
                    help="Layer from which representations are extracted (ignored if Word2Vec is used) for P1 task.")
parser.add_argument("-m", "--llm", dest="llm", type=str, default='bert',
                    help="Masked/Large Language Model (`glove`, `glove-counterfitted`, `bert`, `roberta`, `word2vec`, `fasttext` TODO:{`gpt-2`}).")
parser.add_argument("-sa", "--specify-accuracy", dest="specify_llm_accuracy", type=str, default='',
                    help="Specify the accuracy (in the filename) of a fine-tuned Masked Language Model (this variable is substantially ignored if empty and should not be used on any model but `bert-finetuned`).")
parser.add_argument("-ms", "--llm-size", dest="llm_size", type=str, default='base',
                    help="Type of Masked Language Model (`base`, `large`) (ignored if Word2Vec is used).")
parser.add_argument("-d", "--dataset", dest="dataset", type=str, default='ted',
                    help="Space separated datasets from {`ted`, `ud-english-lines`, `ud-english-pud`, `en-universal`}, e.g., `ted ud-english-lines`.")
parser.add_argument("-cp1", "--classification-p1", dest="classification_p1", type=str, default="True",
                    help="Whether to perform classification or regression on the structural probe task (P1).")
parser.add_argument("-ct1", "--classification-t1", dest="classification_t1", type=str, default="True",
                    help="Whether to perform classification or regression on the POS-tag tasj (T1).")
parser.add_argument("-ml", "--max-length", dest="max_length", type=int, default=20,
                    help="Number of words per-text to use in the analysis (padded/cut if too long/short).")
parser.add_argument("-llm-bs", "--llm-batch-size", dest="llm_batch_size", type=int, default=1024,
                    help="Number of samples processed in batch by an llm (ignored if `llm` is different from an available llm). Reduce it on machines with reduced memory.")
parser.add_argument("-np1", "--num-layers-p1", dest="num_layers_p1", type=int, default=2,
                    help="Number of hidden dense layers for the P1 task(>1).")
parser.add_argument("-nt1", "--num-layers-t1", dest="num_layers_t1", type=int, default=2,
                    help="Number of hidden dense layers for the T1 task (>1).")
parser.add_argument("-nr1", "--num-layers-r1", dest="num_layers_r1", type=int, default=2,
                    help="Number of hidden dense layers for the R1 task (>1).")
parser.add_argument("-nd1", "--num-layers-d1", dest="num_layers_d1", type=int, default=2,
                    help="Number of hidden dense layers for the D1 task (>1).")
parser.add_argument("-t", "--train-split", dest="train_split", type=float, default=0.9,
                    help="Percentage of data reserved to train (the remaining part is used to test).")
parser.add_argument("-ep", "--epochs", dest="epochs", type=int, default=50,
                    help="Training epochs for the each task.")  
parser.add_argument("-s", "--seed", dest="seed", type=int, default=-1,
                    help="Seed.")
parser.add_argument("-b", "--baseline", dest="baseline", type=str, default="False",
                    help="The input X (i.e., the embeddings H) is substituted with a random sample (this can be used to have the performances on random noise).")
parser.add_argument("-k", "--keep-ratio", dest="keep_pos_ratio", type=float, default=0.2,
                    help="Percentage of POS tags kept.")
parser.add_argument("-pb", "--perturbation-budget", dest="perturbation_budget", type=int, default=5,
                    help="Number of words that are perturbed (at most) in each test samples. Take a random value between 1 and this variable's value.")  
parser.add_argument("-bps", "--budget-per-sentence", dest="budget_per_sentence", type=int, default=10,
                    help="Number of sentences perturbed per input sentence, e.g., equal to 2 means that from each sentence we distil two distict perturbed sentences.")  
parser.add_argument("-pm", "--copos", dest="copos", type=str, default="False",
                    help="Whether to substitute words in perturbations with coPOS (coherent Part-of-Speech, i.e., substitutions are consistent).")
parser.add_argument("-wn-mode", "--wordnet-mode", dest="wordnet_mode", type=str, default="True",
                    help="If copos True (otherwise ignored), if True (default), use mode to take the WordNet perturbation, otherwise use least common element.")
parser.add_argument("-pert-scen", "--perturbation-scenario", dest="perturbation_scenario", type=str, default="worst",
                    help="Type of scenario for the robustness measurement (`worst`, `avg`).")
parser.add_argument("-ps", "--probing-size", dest="probing_size", type=int, default=-1,
                    help="Number of samples used to measure robustness (the first `probing_size` of each test set). If -1, the entire test set is considered")  
parser.add_argument("-lp", "--lp-norm", dest="lp_norm", type=int, default=2,
                    help="L_p norm used to calculate the max distance of perturbations in the embedding space.")  

# global variables
args = parser.parse_args()
architecture = str(args.architecture)
activation_p1 = str(args.activation_p1)
activation_t1 = str(args.activation_t1)
activation_r1 = str(args.activation_r1)
activation_d1 = str(args.activation_d1)
layer_llm = int(args.layer_llm)
llm = str(args.llm).lower()
specify_llm_accuracy = str(args.specify_llm_accuracy)  # accuracy of bert fine-tuned model
llm_size = str(args.llm_size)
dataset = list(str(args.dataset).lower().split(' '))
classification_p1 = (True if args.classification_p1.lower()=="true" else False)
classification_t1 = (True if args.classification_t1.lower()=="true" else False)
max_length = int(args.max_length)
llm_batch_size = int(args.llm_batch_size)
num_layers_p1 = int(args.num_layers_p1)
num_layers_t1 = int(args.num_layers_t1)
num_layers_r1 = int(args.num_layers_r1)
num_layers_d1 = int(args.num_layers_d1)
train_split = float(args.train_split)
epochs = int(args.epochs)
seed = int(args.seed)
baseline = (True if args.baseline.lower()=="true" else False)
keep_pos_ratio = float(args.keep_pos_ratio)
perturbation_budget = int(args.perturbation_budget)
budget_per_sentence = int(args.budget_per_sentence)
copos = (True if args.copos.lower()=="true" else False)
perturbation_scenario = str(args.perturbation_scenario)
probing_size = int(args.probing_size)
lp_norm = int(args.lp_norm)
wordnet_mode = (True if args.wordnet_mode.lower()=="true" else False)

# Warning on llm_batch_size
if llm_batch_size > 1:
    print(f"[WARNING] llm_batch_size is equal to {llm_batch_size}, reduce this number if you run out of memory (minimum is 1, but makes the computations slow with llms!).")

if llm == 'bert-finetuned':
    assert len(specify_llm_accuracy) > 0
else:
    assert len(specify_llm_accuracy) == 0

# Set seed
if seed == -1:
    seed = np.random.randint(0, 10000)
print(f"Seed of the session: `{seed}`.")
os.environ['PYTHONHASHSEED']=str(seed)
random.seed(seed)
np.random.seed(seed)
set_seed(seed)

##############################################################################
##############################################################################
########################## Init Operations ###################################
##############################################################################
##############################################################################
# Create the LLM: this will be done only once and used for all the tasks
print(f"Warming-up {llm}...")
if llm in ['bert', 'bert-finetuned']:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=True)
    if llm == 'bert-finetuned':
        bert = BertForMaskedLM.from_pretrained('bert-base-uncased', config=config)
    else:
        bert = BertModel.from_pretrained('bert-base-uncased', config=config)
    # llm mask token id
    mask_id = tokenizer.encode('[MASK]', add_special_tokens=False)[0]
    cls_id = tokenizer.encode('[CLS]', add_special_tokens=False)[0]
    sep_id = tokenizer.encode('[SEP]', add_special_tokens=False)[0]
    pad_id = tokenizer.encode('[PAD]', add_special_tokens=False)[0]
    pad_token = '[PAD]'
    if llm == 'bert-finetuned':
        bert_weights_path = f'./../../../data/models/bert-finetuned/bert_pretrained_sst_saved_weights_inputlen-20_accuracy-{specify_llm_accuracy}.pt'
        print(f"Loading pre-saved weights at `{bert_weights_path}`")
        bert = SentimentalBERT(bert, max_length, llm=True)
        bert.load_state_dict(torch.load(bert_weights_path))
    # Freeze the parameters
    for param in bert.parameters():
        param.requires_grad = False

elif llm == 'roberta':
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    config = RobertaConfig.from_pretrained("roberta-base", output_hidden_states=True, output_attention=True)
    roberta = RobertaModel.from_pretrained('roberta-base', config=config)
    # llm mask token id
    mask_id = tokenizer.encode('<mask>', add_special_tokens=False)[0]
    cls_id = tokenizer.encode('<s>', add_special_tokens=False)[0]
    sep_id = tokenizer.encode('</s>', add_special_tokens=False)[0]
    pad_id = tokenizer.encode('<pad>', add_special_tokens=False)[0]
    pad_token = '<pad>'
    # Freeze the parameters
    for param in roberta.parameters():
        param.requires_grad = False

elif llm == 'glove':
    embedding_filename = './../../../data/embeddings/glove.840B.300d.txt'
    word2toindex, index2word, index2embedding = load_embedding(embedding_filename, try_first='./../../../data/embeddings/')

elif llm == 'glove-counterfitted':
    embedding_filename = './../../../data/embeddings/glove-counterfitted.840B.300d.txt'
    word2toindex, index2word, index2embedding = load_embedding(embedding_filename)

elif llm == 'word2vec':
    embedding_filename = './../../../data/embeddings/google.news.300d.txt'
    word2toindex, index2word, index2embedding = load_embedding(embedding_filename)

elif llm == 'fasttext':
    embedding_filename = './../../../data/embeddings/fasttext.en.300d.txt'
    word2toindex, index2word, index2embedding = load_embedding(embedding_filename)

elif llm == 'gpt-2':
    pass
    
else:
    raise Exception(f"{llm} is not a valid value for llm.")

# Sorting prevents from saving permutations of the same datasets
dataset.sort()  

##############################################################################
##############################################################################
########################## P1-structural probe ###############################
##############################################################################
##############################################################################
# Get the sentences and the distance matrices (possibly from pre-saved files)
print(f"Collecting texts and distance matrices from `{dataset}` dataset...")
save_prefix = f"./../../../data/datasets/conll/{dataset}-{llm}/"
os.makedirs(save_prefix, exist_ok=True)
S_file = Path(save_prefix+f"S_maxlen-{max_length}{'' if llm!='bert-finetuned' else f'_acc-{specify_llm_accuracy}'}.npy")
M_file = Path(save_prefix+f"M_maxlen-{max_length}{'' if llm!='bert-finetuned' else f'_acc-{specify_llm_accuracy}'}.npy")
if S_file.is_file() and M_file.is_file():
    S = np.load(str(S_file), allow_pickle=True)
    M = np.load(str(M_file), allow_pickle=True)
else:
    S, M = get_trees_from_conlldatasets(dataset, maxlen=max_length, nsamples=-1)
    M[M==999] = 0.
    # Save
    np.save(str(S_file), S)
    np.save(str(M_file), M)

# Collect the input representations
# This will be done only once and used for all the tasks (but it cannot be moved to the prev. section as we need S to be available)
print(f"Collecting the {llm} texts representations...")
suffix_layer = (f'_layer-{[layer_llm]}' if llm in ['bert', 'roberta', 'bert-finetuned'] else '')
H_pathname = save_prefix+f"H_maxlen-{max_length}{suffix_layer}{'' if llm!='bert-finetuned' else f'_acc-{specify_llm_accuracy}'}.npy"
H_file = Path(H_pathname)
if H_file.is_file():
    X = np.load(H_file._str, allow_pickle=True)
    embedding_dim =  X.shape[1]
    trailing_dims = X.shape[1:]
else:
    X = []
    if llm in ['word2vec', 'glove', 'glove-counterfitted', 'fasttext']:
        embedding_dim = 300
        trailing_dims = embedding_dim*max_length
        embed = lambda w: index2embedding[word2toindex[w]]
        zeros = np.zeros((embedding_dim,))
        for s in tqdm.tqdm(S):
            s_split = s.split(' ')
            X += [np.array([[embed(w) for w in s_split] + [zeros]*(max_length-len(s_split))]).reshape(1,embedding_dim*max_length)]
        X = np.array(X).reshape(len(X), embedding_dim, max_length)

    else:
        # Create bert inputs and masks
        for batch_range in tqdm.tqdm(range(0, len(S), llm_batch_size)):
            BERT_INPUTS, INPUT_MASKS = [], []
            for s in tqdm.tqdm(S[batch_range:batch_range+llm_batch_size]):
                s_split = s.split(' ')
                s_pad = s_split + [pad_token]*(max_length-len(s_split))
                bert_input = torch.tensor(tokenizer.convert_tokens_to_ids(s_pad)).reshape(1,-1)
                input_mask = torch.tensor([1 for _ in s_split]+[0 for _ in range(max_length-len(s_split))]).reshape(1,-1)
                BERT_INPUTS += [bert_input]
                INPUT_MASKS += [input_mask]
            BERT_INPUTS = torch.tensor([b.numpy() for b in BERT_INPUTS]).reshape(len(BERT_INPUTS), max_length)
            INPUT_MASKS = torch.tensor([i.numpy() for i in INPUT_MASKS]).reshape(len(INPUT_MASKS), max_length)
            if llm == 'bert':
                x_p1 = bert(BERT_INPUTS, attention_mask=INPUT_MASKS)[2][layer_llm][:,:,:] # collect all the tokens, then average
            elif llm == 'bert-finetuned':
                x_p1 = bert(BERT_INPUTS, mask=INPUT_MASKS)[1][layer_llm][:,:,:] # collect all the tokens, then average
            elif llm == 'roberta':
                x_p1 = roberta(BERT_INPUTS, attention_mask=INPUT_MASKS)[2][layer_llm][:,:,:]
            else: # gpt
                raise Exception(f"{llm} is not a valid masked language model.")
            X += [[x] for x in x_p1]
        embedding_dim =  x_p1.shape[2]
        trailing_dims = x_p1.shape[1:]
        X = np.array([x[0].numpy() for x in X]).reshape(len(X), embedding_dim, max_length)
    # Store the file
    np.save(str(H_file), X)

# Train-test split
if architecture == 'fc':
    dataset_shape = (len(X), np.prod(trailing_dims)) 
elif architecture in ['cnn', 'rnn']:
    dataset_shape = (len(X), embedding_dim, max_length)
elif architecture == 'cnn2d':
    dataset_shape = (len(X), embedding_dim, max_length, 1)
else:
    raise Exception(f"{architecture} is not a valid architecture")

# Check the `baseline` var
if baseline is True:
    print("\n[WARNING] Baseline mode is activated, X (i.e., the embeddings H) are replaced with random noise!\n")
    X = np.random.rand(*dataset_shape)
else:
    X = X.reshape(*dataset_shape)

# Input matrices and train-test split
if classification_p1 is True:
    n_pos = int(np.max(M)+1)  # number of classes in the classification setting
    M = to_categorical(M)
    M = M.reshape(len(M), max_length*max_length, n_pos)
else:
    M = M.reshape(len(M), max_length*max_length)
n_train, n_test = int(train_split*len(X)), int((1.-train_split)*len(X))
X_train, M_train = X[:n_train], M[:n_train]
X_test, M_test = X[n_train:], M[n_train:]
assert len(X_train) == len(M_train) and len(X_test) == len(M_test)

# Create the model
input_shape = X[0].shape
distances_shape = M[0].shape
output_shape = (max_length*max_length if classification_p1 is False else (max_length*max_length, n_pos))
P1_model = Sequential()
if architecture == 'fc':
    P1_model.add(Dense(512, input_shape=input_shape, activation=activation_p1))
    P1_model.add(Dropout(0.3))
elif architecture == 'cnn':
    P1_model.add(Conv1D(256, kernel_size=embedding_dim, strides=embedding_dim, input_shape=input_shape, activation=activation_p1))
elif architecture == 'cnn2d':
    P1_model.add(Conv2D(256, kernel_size=(embedding_dim, 2), strides=(1, 1), input_shape=input_shape, activation=activation_p1))
    P1_model.add(Flatten())
elif architecture == 'rnn':
    P1_model.add(Bidirectional(LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2), input_shape=input_shape))
    P1_model.add(Flatten())
    #model.add(Bidirectional(LSTM(256)))
else:
    raise Exception(f"{architecture} is not a valid architecture")

# Add the deeper layers
for _ in range(num_layers_p1-1):
    P1_model.add(Dense(512, activation=activation_p1))
    P1_model.add(Dropout(0.3))
# output layer
if classification_p1 is True:
    P1_model.add(Dense(np.prod(output_shape), activation=p1_custom_softmax))
    P1_model.add(Reshape(output_shape))
    P1_model.compile(optimizer='adam', metrics=['mae', 'binary_crossentropy'], loss='binary_crossentropy')
else:
    P1_model.add(Dense(output_shape, activation='linear'))
    P1_model.compile(optimizer='adam', metrics=['mae', 'mse'], loss='mae')

# Train
callback_earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4, restore_best_weights=True)
callback_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
h = P1_model.fit(X_train, M_train,
        batch_size=512,
        epochs=epochs,
        verbose=1,
        validation_data=(X_test, M_test),
        callbacks=[callback_earlystop, callback_scheduler]
        )

# Test
print(f"\nTest set evaluation:")
p = P1_model.evaluate(X_test, M_test)

if architecture == 'fc':
    M_hat = P1_model.predict(X_test.reshape(len(X_test), np.prod(trailing_dims)))
elif architecture == 'cnn':
    M_hat = P1_model.predict(X_test.reshape(len(X_test), embedding_dim, max_length)).squeeze(1)
elif architecture == 'cnn2d':
    M_hat = P1_model.predict(X_test.reshape(len(X_test), embedding_dim, max_length, 1)).squeeze(1)
elif architecture == 'rnn':
    M_hat = P1_model.predict(X_test.reshape(len(X_test), embedding_dim, max_length))
else:
    raise Exception(f"{architecture} is not a valid architecture")

if classification_p1 is True:
    M = np.argmax(M, axis=2).reshape(len(M), max_length, max_length)
    M_hat = np.argmax(M_hat, axis=2).reshape(len(X_test), max_length, max_length)
    M_test = np.argmax(M_test, axis=2).reshape(len(X_test), max_length, max_length)

sm = same_distance(M_test, M_hat, max_length)
sr_pairwise = spearman_pairwise(M_test, M_hat, max_length)
sr = spearman(M_test, M_hat, max_length)
sr_adj = spearman(M_test, M_hat, max_length, adjacency=True)
uuas = UUAS(M_test, M_hat, max_length)
print(f"SDR: {sm}, UUAS: {uuas}, Spearman: {sr}\n")

##############################################################################
##############################################################################
########################## T1-POS tagging ####################################
##############################################################################
##############################################################################
# Baseline?
if baseline is True:
    print("\n[WARNING] Baseline mode is activated, X (i.e., the embeddings H) are replaced with random noise!\n")

# Get the sentences and the distance matrices (only from pre-saved files)
print(f"Collecting texts and POS tags from `{dataset}` dataset...")
POS_file = Path(save_prefix+f"POS_maxlen-{max_length}_keep-POS-ratio-{keep_pos_ratio}{'' if llm!='bert-finetuned' else f'_acc-{specify_llm_accuracy}'}.npy")
if POS_file.is_file():
    POS = np.load(str(POS_file), allow_pickle=True)
else:
    _, _, _, POS = get_pos_from_conlldatasets(dataset, nsamples=-1, filter_ratio=keep_pos_ratio)
    # Cut max length
    for i,p in zip(range(len(POS)), POS):
        POS[i] = POS[i][:max_length]
        # pad the POS
        POS[i] += [0]*(max_length-len(POS[i]))
    POS = np.array(POS)
    # Save
    np.save(str(POS_file), POS)

# Train-test split
if architecture == 'fc':
    dataset_shape = (len(X), np.prod(trailing_dims)) 
elif architecture in ['cnn', 'rnn']:
    dataset_shape = (len(X), embedding_dim, max_length)
elif architecture == 'cnn2d':
    dataset_shape = (len(X), embedding_dim, max_length, 1)
else:
    raise Exception(f"{architecture} is not a valid architecture")

# Check the `baseline` var
if baseline is True:
    print("\n[WARNING] Baseline mode is activated, X (i.e., the embeddings H) are replaced with random noise!\n")
    X = np.random.rand(*dataset_shape)
else:
    X = X.reshape(*dataset_shape)

# Input matrices and train-test split
n_pos = int(np.max(POS))+1
POS = POS.reshape(len(POS), max_length)
if classification_t1 is True:
    POS = to_categorical(POS)
n_train, n_test = int(train_split*len(X)), int((1.-train_split)*len(X))
X_train, POS_train = X[:n_train], POS[:n_train]
X_test, POS_test = X[n_train:], POS[n_train:]
assert len(X_train) == len(POS_train) and len(X_test) == len(POS_test)

# Create the model
input_shape = X[0].shape
distances_shape = POS[0].shape
output_shape = (max_length,n_pos)
T1_model = Sequential()
if architecture == 'fc':
    T1_model.add(Dense(512, input_shape=input_shape, activation=activation_t1))
    T1_model.add(Dropout(0.3))
elif architecture == 'cnn':
    T1_model.add(Conv1D(256, kernel_size=embedding_dim, strides=embedding_dim, input_shape=input_shape, activation=activation_t1))
elif architecture == 'cnn2d':
    T1_model.add(Conv2D(256, kernel_size=(embedding_dim, 2), strides=(1, 1), input_shape=input_shape, activation=activation_t1))
    T1_model.add(Flatten())
elif architecture == 'rnn':
    T1_model.add(Bidirectional(LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2), input_shape=input_shape))
    T1_model.add(Flatten())
    #model.add(Bidirectional(LSTM(256)))
else:
    raise Exception(f"{architecture} is not a valid architecture")

# Add the deeper layers
for _ in range(num_layers_t1-1):
    T1_model.add(Dense(512, activation=activation_t1))
    T1_model.add(Dropout(0.3))
# output layer
if classification_t1 is True:
    T1_model.add(Dense(np.prod(output_shape), activation=t1_custom_softmax))
    T1_model.add(Reshape(output_shape))
    T1_model.compile(optimizer='adam', metrics=['mae', 'mse', 'binary_crossentropy'], loss='binary_crossentropy')
else:
    T1_model.add(Dense(max_length, activation='linear'))
    T1_model.compile(optimizer='adam', metrics=['mae', 'mse', 'categorical_crossentropy'], loss='mae')

# Train
callback_earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4, restore_best_weights=True)
callback_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
h = T1_model.fit(X_train, POS_train,
        batch_size=512,
        epochs=epochs,
        verbose=1,
        validation_data=(X_test, POS_test), 
        callbacks=[callback_earlystop, callback_scheduler]
        )

# Predict on the entire test set
if architecture == 'fc':
    POS_hat = T1_model.predict(X_test.reshape(len(X_test), np.prod(trailing_dims)))
elif architecture == 'cnn':
    POS_hat = T1_model.predict(X_test.reshape(len(X_test), embedding_dim, max_length)).squeeze(1)
elif architecture == 'cnn2d':
    POS_hat = T1_model.predict(X_test.reshape(len(X_test), embedding_dim, max_length, 1)).squeeze(1)
elif architecture == 'rnn':
    POS_hat = T1_model.predict(X_test.reshape(len(X_test), embedding_dim, max_length))
else:
    raise Exception(f"{architecture} is not a valid architecture")

# Test
print(f"\nTest set evaluation:")
p = T1_model.evaluate(X_test, POS_test)

accuracy_t1 = t1_accuracy(POS_test, POS_hat, classification=classification_t1)
print(f"\n T1. Accuracy: {accuracy_t1}\n")

##############################################################################
##############################################################################
################### R1-Root Tree-root identification #########################
##############################################################################
##############################################################################
# Baseline?
if baseline is True:
    print("\n[WARNING] Baseline mode is activated, X (i.e., the embeddings H) are replaced with random noise!\n")

# Get the sentences and the distance matrices (possibly from pre-saved files)
print(f"Collecting texts and distance matrices from `{dataset}` dataset...")
os.makedirs(save_prefix, exist_ok=True)  # create the folder in case it doesn't exists
R_file = Path(save_prefix+f"R_maxlen-{max_length}{'' if llm!='bert-finetuned' else f'_acc-{specify_llm_accuracy}'}.npy")
if R_file.is_file():
    print(f"Loading files from {R_file} path.")
    R = np.load(str(R_file), allow_pickle=True)
else:
    _, R = get_roots_from_conlldatasets(dataset, maxlen=max_length, nsamples=-1)  # please notice that S is unused but kept for compatibility
    assert len(R) == len(S)
    # Save
    print(f"Saving files to {R_file} path.")
    np.save(str(R_file), R)
    
# Train-test split
if architecture == 'fc':
    dataset_shape = (len(X), np.prod(trailing_dims)) 
elif architecture in ['cnn', 'rnn']:
    dataset_shape = (len(X), embedding_dim, max_length)
elif architecture == 'cnn2d':
    dataset_shape = (len(X), embedding_dim, max_length, 1)
else:
    raise Exception(f"{architecture} is not a valid architecture")

# Check the `baseline` var
if baseline is True:
    print("\n[WARNING] Baseline mode is activated, X (i.e., the embeddings H) are replaced with random noise!\n")
    X = np.random.rand(*dataset_shape)
else:
    X = X.reshape(*dataset_shape)

n_train, n_test = int(train_split*len(X)), int((1.-train_split)*len(X))
X_train, R_train = X[:n_train], R[:n_train]
X_test, R_test = X[n_train:], R[n_train:]
assert len(X_train) == len(R_train) and len(X_test) == len(R_test)

# Create the model
input_shape = X[0].shape
distances_shape = R[0].shape
output_shape = max_length
R1_model = Sequential()
if architecture == 'fc':
    R1_model.add(Dense(512, input_shape=input_shape, activation=activation_r1))
    R1_model.add(Dropout(0.3))
elif architecture == 'cnn':
    R1_model.add(Conv1D(256, kernel_size=embedding_dim, strides=embedding_dim, input_shape=input_shape, activation=activation_r1))
elif architecture == 'cnn2d':
    R1_model.add(Conv2D(256, kernel_size=(embedding_dim, 2), strides=(1, 1), input_shape=input_shape, activation=activation_r1))
    R1_model.add(Flatten())
elif architecture == 'rnn':
    R1_model.add(Bidirectional(LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2), input_shape=input_shape,))
    R1_model.add(Flatten())
    #R1_model.add(Bidirectional(LSTM(256, dropout=0.2, recurrent_dropout=0.2)))
else:
    raise Exception(f"{architecture} is not a valid architecture")

# Add the deeper layers
for _ in range(num_layers_r1-1):
    R1_model.add(Dense(512, activation=activation_r1))
    R1_model.add(Dropout(0.3))
# output layer
R1_model.add(Dense(output_shape, activation="softmax"))
R1_model.compile(optimizer='adam', metrics=['mae', 'categorical_crossentropy'], loss='categorical_crossentropy')

# Train
callback_earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4, restore_best_weights=True)
callback_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
h = R1_model.fit(X_train, R_train,
        batch_size=512,
        epochs=epochs,
        verbose=1,
        validation_data=(X_test, R_test),
        callbacks=[callback_earlystop, callback_scheduler]
        )

# Predict on the entire test set
if architecture == 'fc':
    R_hat = R1_model.predict(X_test.reshape(len(X_test), np.prod(trailing_dims)))
elif architecture == 'cnn':
    R_hat = R1_model.predict(X_test.reshape(len(X_test), embedding_dim, max_length)).squeeze(1)
elif architecture == 'cnn2d':
    R_hat = R1_model.predict(X_test.reshape(len(X_test), embedding_dim, max_length, 1))
elif architecture == 'rnn':
    R_hat = R1_model.predict(X_test.reshape(len(X_test), embedding_dim, max_length))
else:
    raise Exception(f"{architecture} is not a valid architecture")

accuracy_r1 = r1_accuracy(R_test, R_hat)

print(f"Dataset(s): {dataset}")
print(f"Accuracy over {len(R_test)} samples: {accuracy_r1}\n")

##############################################################################
##############################################################################
################### D1-Root Depth of the tree ################################
##############################################################################
##############################################################################
# Baseline?
if baseline is True:
    print("\n[WARNING] Baseline mode is activated, X (i.e., the embeddings H) are replaced with random noise!\n")

# Get the sentences and the distance matrices (possibly from pre-saved files)
print(f"Collecting texts and distance matrices from `{dataset}` dataset...")
os.makedirs(save_prefix, exist_ok=True)  # create the folder in case it doesn't exists
D_file = Path(save_prefix+f"D_maxlen-{max_length}{'' if llm!='bert-finetuned' else f'_acc-{specify_llm_accuracy}'}.npy")
if D_file.is_file():
    print(f"Loading files from {D_file}.")
    D = np.load(str(D_file), allow_pickle=True)
else:
    _, D = get_depths_from_conlldatasets(dataset, maxlen=max_length, nsamples=-1)  # please notice that S is unused but kept for compatibility
    # Save
    print(f"Saving files to {D_file}.")
    np.save(str(D_file), D)

# Train-test split
if architecture == 'fc':
    dataset_shape = (len(X), np.prod(trailing_dims)) 
elif architecture in ['cnn', 'rnn']:
    dataset_shape = (len(X), embedding_dim, max_length)
elif architecture == 'cnn2d':
    dataset_shape = (len(X), embedding_dim, max_length, 1)
else:
    raise Exception(f"{architecture} is not a valid architecture")

# Check the `baseline` var
if baseline is True:
    print("\n[WARNING] Baseline mode is activated, X (i.e., the embeddings H) are replaced with random noise!\n")
    X = np.random.rand(*dataset_shape)
else:
    X = X.reshape(*dataset_shape)

n_train, n_test = int(train_split*len(X)), int((1.-train_split)*len(X))
X_train, D_train = X[:n_train], D[:n_train]
X_test, D_test = X[n_train:], D[n_train:]
assert len(X_train) == len(D_train) and len(X_test) == len(D_test)

# Create the model
input_shape = X[0].shape
distances_shape = D[0].shape
output_shape = max_length
D1_model = Sequential()
if architecture == 'fc':
    D1_model.add(Dense(512, input_shape=input_shape, activation=activation_d1))
    D1_model.add(Dropout(0.3))
elif architecture == 'cnn':
    D1_model.add(Conv1D(256, kernel_size=embedding_dim, strides=embedding_dim, input_shape=input_shape, activation=activation_d1))
elif architecture == 'cnn2d':
    D1_model.add(Conv2D(256, kernel_size=(embedding_dim, 2), strides=(1, 1), input_shape=input_shape, activation=activation_d1))
    D1_model.add(Flatten())
elif architecture == 'rnn':
    D1_model.add(Bidirectional(LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2), input_shape=input_shape))
    model.add(Flatten())
    #D1_model.add(Bidirectional(LSTM(256, dropout=0.2, recurrent_dropout=0.2)))
else:
    raise Exception(f"{architecture} is not a valid architecture")

# Add the deeper layers
for _ in range(num_layers_d1-1):
    D1_model.add(Dense(512, activation=activation_d1))
    D1_model.add(Dropout(0.3))
# output layer
D1_model.add(Dense(output_shape, activation="softmax"))
D1_model.compile(optimizer='adam', metrics=['binary_crossentropy', 'categorical_crossentropy'], loss='categorical_crossentropy')

# Train
callback_earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4, restore_best_weights=True)
callback_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
h = D1_model.fit(X_train, D_train,
                batch_size=512,
                epochs=epochs,
                verbose=1,
                validation_data=(X_test, D_test),
                callbacks=[callback_earlystop, callback_scheduler]
                )

if architecture == 'fc':
    D_hat = D1_model.predict(X_test.reshape(len(X_test), np.prod(trailing_dims)))
elif architecture == 'cnn':
    D_hat = D1_model.predict(X_test.reshape(len(X_test), embedding_dim, max_length)).squeeze(1)
elif architecture == 'cnn2d':
    D_hat = D1_model.predict(X_test.reshape(len(X_test), embedding_dim, max_length, 1))
elif architecture == 'rnn':
    D_hat = D1_model.predict(X_test.reshape(len(X_test), embedding_dim, max_length))
else:
    raise Exception(f"{architecture} is not a valid architecture")

sd_ratio_d1 = d1_same_distance(D_test, D_hat)
sp_ratio_d1 = d1_spearman(D_test, D_hat)
print(f"Dataset(s): {dataset}")
print(f"On {len(D_test)} samples: same distance ratio: {sd_ratio_d1}, Spearman correlation: {sp_ratio_d1}\n")

##############################################################################
##############################################################################
########################## Robustness on P1,T1, R1 ###########################
##############################################################################
##############################################################################
# Extract the perturbed samples
perturbation_config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
perturbation_model = BertForMaskedLM.from_pretrained('bert-base-uncased', config=perturbation_config).to('cpu')
perturbation_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
bps = (1 if perturbation_scenario == 'avg' else budget_per_sentence)
S_prime = []
M_test_touse, POS_test_touse, R_test_touse, D_test_touse = [], [], [], []  # used to collect repeated *clean* test labels
S_test = S[n_train:]
print(f"Collecting representations of perturbed inputs...")
if probing_size == -1:
    probing_size = n_test
# Isolate a random permutation of the test set (to be perturbed)
for idx_s, s in tqdm.tqdm(enumerate(S_test)):
    s_split = s.split(' ')
    for _ in range(bps):
        tmp_s_prime = cp.copy(s_split)
        sampled_perturbation_budget = random.randint(1, perturbation_budget)  # random.randint(a,b) samples in {a,b}, while np.random.randint from {a, b-1} :(
        interventions = np.random.permutation(len(s_split))[:sampled_perturbation_budget]
        for i in interventions:
            if copos is False:
                # Use BERT LLM to predict perturbations (no coPOS)
                masked_text = cp.copy(s_split)
                masked_text[i] = '[MASK]'
                s_prime = sequential_interventions(perturbation_model, masked_text, [[[i]]], combine_subs=True, topk=1, budget=1, device='cpu', TOKENIZER=perturbation_tokenizer)
                tmp_s_prime[i] = s_prime[0][i]
            else:  # Use WordNet 
                if wordnet_mode is True:  #(coPOS via majority-vote)
                    tmp_s_prime[i] = wordnet_perturbations(tmp_s_prime, i, topk=10, least_common=False)
                else:  #(coPOS via minority-vote)
                    tmp_s_prime[i] = wordnet_perturbations(tmp_s_prime, i, topk=10, least_common=True)    
        S_prime += [tmp_s_prime]
        # Collect the test labels
        M_test_touse.append(M_test[idx_s])
        POS_test_touse.append(POS_test[idx_s])
        R_test_touse.append(R_test[idx_s])
        D_test_touse.append(D_test[idx_s])

assert len(S_prime) == len(M_test_touse) == len(POS_test_touse) == len(R_test_touse) == len(D_test_touse)

print("Collecting the representations from the embeddings...")
X_prime = []
if llm in ['word2vec', 'glove', 'glove-counterfitted', 'fasttext']:
    embedding_dim = 300
    trailing_dims = embedding_dim*max_length
    embed = lambda w: index2embedding[word2toindex[w]]
    zeros = np.zeros((embedding_dim,))
    for s_split in tqdm.tqdm(S_prime):
        X_prime += [np.array([[embed(w) for w in s_split] + [zeros]*(max_length-len(s_split))]).reshape(1,embedding_dim*max_length)]
    X_prime = np.array(X_prime).reshape(len(X_prime), embedding_dim, max_length)

else:
    # Create bert inputs and masks
    for batch_range in tqdm.tqdm(range(0, len(S_prime), llm_batch_size)):
        BERT_INPUTS, INPUT_MASKS = [], []
        for s_split in tqdm.tqdm(S_prime[batch_range:batch_range+llm_batch_size]):
            s_pad = s_split + [pad_token]*(max_length-len(s_split))
            bert_input = torch.tensor(tokenizer.convert_tokens_to_ids(s_pad)).reshape(1,-1)
            input_mask = torch.tensor([1 for _ in s_split]+[0 for _ in range(max_length-len(s_split))]).reshape(1,-1)
            BERT_INPUTS += [bert_input]
            INPUT_MASKS += [input_mask]
        BERT_INPUTS = torch.tensor([b.numpy() for b in BERT_INPUTS]).reshape(len(BERT_INPUTS), max_length)
        INPUT_MASKS = torch.tensor([i.numpy() for i in INPUT_MASKS]).reshape(len(INPUT_MASKS), max_length)
        if llm == 'bert':
            x_p1 = bert(BERT_INPUTS, attention_mask=INPUT_MASKS)[2][layer_llm][:,:,:] # collect all the tokens, then average
        elif llm == 'bert-finetuned':
            x_p1 = bert(BERT_INPUTS, mask=INPUT_MASKS)[1][layer_llm][:,:,:] # collect all the tokens, then average
        elif llm == 'roberta':
            x_p1 = roberta(BERT_INPUTS, attention_mask=INPUT_MASKS)[2][layer_llm][:,:,:]
        else: # gpt
            raise Exception(f"{llm} is not a valid masked language model.")
        X_prime += [[x] for x in x_p1]
    embedding_dim =  x_p1.shape[2]
    trailing_dims = x_p1.shape[1:]
    X_prime = np.array([x[0].numpy() for x in X_prime]).reshape(len(X_prime), embedding_dim, max_length)

# Average L_p distance and cosine-similarity of the farthest, perturbed inputs
max_norm = []  # in p-norms the wors case is max-distance
min_cosine = []  # we use cosine similarity, so worst case is min-similarity
for i,x in enumerate(X_test[:probing_size]):
    tmp_max_norm = []
    tmp_min_cosine = []
    for idx_bps in range(bps):
        x_prime_p1 = X_prime[(i*budget_per_sentence)+idx_bps].flatten()
        tmp_max_norm += [np.linalg.norm(x-x_prime_p1, ord=lp_norm)]
        tmp_min_cosine += [1 - spatial.distance.cosine(x, x_prime_p1)]
    if perturbation_scenario == 'avg':
        max_norm += [np.mean(tmp_max_norm)]
        min_cosine += [np.mean(tmp_min_cosine)]
    else: # `worst`
        max_norm += [max(tmp_max_norm)]
        min_cosine += [min(tmp_min_cosine)]
lp_robustness = np.mean(max_norm)
lp_robustness_perdim = lp_robustness/embedding_dim
cosine_robustness = np.mean(min_cosine)

# P1 and T1 robustness
print(f"Testing performances of the models on P1, T1 and R1")
if architecture == 'fc':
    M_hat_prime = P1_model.predict(X_prime.reshape(len(X_prime), np.prod(trailing_dims)))
    POS_hat_prime = T1_model.predict(X_prime.reshape(len(X_prime), np.prod(trailing_dims)))
    Root_hat_prime = R1_model.predict(X_prime.reshape(len(X_prime), np.prod(trailing_dims)))
    Depth_hat_prime = D1_model.predict(X_prime.reshape(len(X_prime), np.prod(trailing_dims)))
elif architecture == 'cnn':
    M_hat_prime = P1_model.predict(X_prime.reshape(len(X_prime), embedding_dim, max_length)).squeeze(1)
    POS_hat_prime = T1_model.predict(X_prime.reshape(len(X_prime), embedding_dim, max_length)).squeeze(1)
    Root_hat_prime = R1_model.predict(X_prime.reshape(len(X_prime), embedding_dim, max_length)).squeeze(1)
    Depth_hat_prime = D1_model.predict(X_prime.reshape(len(X_prime), embedding_dim, max_length)).squeeze(1)
elif architecture == 'cnn2d':
    M_hat_prime = P1_model.predict(X_prime.reshape(len(X_prime), embedding_dim, max_length, 1)).squeeze(1)
    POS_hat_prime = T1_model.predict(X_prime.reshape(len(X_prime), embedding_dim, max_length, 1)).squeeze(1)
    Root_hat_prime = R1_model.predict(X_prime.reshape(len(X_prime), embedding_dim, max_length, 1)).squeeze(1)
    Depth_hat_prime = D1_model.predict(X_prime.reshape(len(X_prime), embedding_dim, max_length, 1)).squeeze(1)
elif architecture == 'rnn':
    M_hat_prime = P1_model.predict(X_prime.reshape(len(X_prime), embedding_dim, max_length))
    POS_hat_prime = T1_model.predict(X_prime.reshape(len(X_prime), embedding_dim, max_length))
    Root_hat_prime = R1_model.predict(X_prime.reshape(len(X_prime), embedding_dim, max_length))
    Depth_hat_prime = D1_model.predict(X_prime.reshape(len(X_prime), embedding_dim, max_length))
else:
    raise Exception(f"{architecture} is not a valid architecture")

if classification_p1 is True:
    M_hat_prime = np.argmax(M_hat_prime, axis=2).reshape(len(M_hat_prime), max_length, max_length)

# Cut M_test and POS_test to the size of the predictions
print("Calculating metrics on perturbed inputs...")
D_test_touse = np.array(D_test_touse)
if perturbation_scenario == 'avg':
    # P1
    sm_prime = same_distance(M_test_touse, M_hat_prime, max_length)
    uuas_prime = UUAS(M_test_touse, M_hat_prime, max_length)
    # T1
    accuracy_prime_t1 = t1_accuracy(POS_test_touse, POS_hat_prime, classification=classification_t1)
    # R1
    accuracy_prime_r1 = r1_accuracy(R_test_touse, Root_hat_prime)
    # D1
    sd_ratio_prime_d1 = d1_same_distance(D_test_touse, Depth_hat_prime)
else:  # `worst`-case scenario
    sm_prime, sr_prime, uuas_prime = [], [], []
    accuracy_prime_t1 = []
    accuracy_prime_r1 = []
    sd_ratio_prime_d1 = []
    for i in range(min(len(M_test), probing_size)):
        sm_prime_worst, uuas_prime_worst = [], []
        accuracy_prime_t1_worst = []
        accuracy_prime_r1_worst = []
        sm_prime_d1_worst, sp_prime_d1_worst = [], []
        for idx_bps in range(bps):
            # P1
            m_test = M_test_touse[i].reshape(1,-1)
            m_hat_prime = M_hat_prime[(i*budget_per_sentence)+idx_bps].reshape(1, -1)
            sm_prime_worst += [same_distance(m_test, m_hat_prime, max_length)]
            uuas_prime_worst += [UUAS(m_test, m_hat_prime, max_length)]
            # T1
            accuracy_prime_t1_worst += [t1_accuracy(POS_test_touse[i], POS_hat_prime[(i*budget_per_sentence)+idx_bps], classification=classification_t1)]
            # R1
            accuracy_prime_r1_worst += [r1_accuracy(R_test_touse[i].reshape(1,-1), Root_hat_prime[(i*budget_per_sentence)+idx_bps].reshape(1,-1))]
            # D1
            d_test = D_test_touse[i].reshape(1,-1)
            d_hat = Depth_hat_prime[(i*budget_per_sentence)+idx_bps].reshape(1,-1)
            sm_prime_d1_worst += [d1_same_distance(d_test, d_hat)]
        sm_prime += [min(sm_prime_worst)]
        uuas_prime += [min(uuas_prime_worst)]
        accuracy_prime_t1 += [min(accuracy_prime_t1_worst)]
        accuracy_prime_r1 += [min(accuracy_prime_r1_worst)]
        sd_ratio_prime_d1 += [min(sm_prime_d1_worst)]

    # Spearman requires a sequence, hence avg correponds to worst
    sr_prime = spearman(M_test_touse, M_hat_prime, max_length)
    sp_ratio_prime_d1 = d1_spearman(D_test_touse, Depth_hat_prime)
    # Average the other metrics
    sm_prime, uuas_prime =  np.mean(sm_prime), np.mean(uuas_prime)
    accuracy_prime_t1 = np.mean(accuracy_prime_t1)
    accuracy_prime_r1 = np.mean(accuracy_prime_r1)
    same_distance_prime_d1 =  np.mean(sd_ratio_prime_d1)

##############################################################################
##############################################################################
########################## Summary P1+T1 #####################################
##############################################################################
##############################################################################
P1_string_list, T1_string_list, R1_string_list, D1_string_list = [], [], [], []
P1_model.summary(line_length=80, print_fn=lambda x: P1_string_list.append(x))
P1_string_list = '\n'.join(P1_string_list)
T1_model.summary(line_length=80, print_fn=lambda x: T1_string_list.append(x))
T1_string_list = '\n'.join(T1_string_list)
R1_model.summary(line_length=80, print_fn=lambda x: R1_string_list.append(x))
R1_string_list = '\n'.join(R1_string_list)
D1_model.summary(line_length=80, print_fn=lambda x: D1_string_list.append(x))
D1_string_list = '\n'.join(D1_string_list)
os.makedirs('./results/robustness/', exist_ok=True)
baseline_suffix = ('' if baseline is False else '_BASELINE-random-inputs')
with open(f'./results/robustness/{dataset}_{llm}{baseline_suffix}.txt', 'a+') as file_:
    file_.write(f"{'#'*100}\n")
    file_.write(f"Seed: {seed}\n")
    file_.write(f"model: {architecture}, embeddings: {llm}\n")
    if len(specify_llm_accuracy) > 0:
        file_.write(f"Fine-tuned accuracy {specify_llm_accuracy}\n")
    file_.write(f"Layer P1: {layer_llm}, layer T1: {layer_llm}\n")
    file_.write(f"Dataset(s): {dataset}, input-length: {max_length}, keep_pos_ratio: {keep_pos_ratio}\n")
    file_.write(f"Classification_p1: {classification_p1}, classification_t1: {classification_t1}, classification_r1: true\n")
    file_.write(f"Training epochs: P1: {epochs}, T1: {epochs}, R1: {epochs}\n")
    file_.write(f"P1 MODEL\n")
    file_.write(f"{P1_string_list}\n")
    file_.write(f"T1 MODEL\n")
    file_.write(f"{T1_string_list}\n")
    file_.write(f"R1 MODEL\n")
    file_.write(f"{R1_string_list}\n")
    file_.write(f"D1 MODEL\n")
    file_.write(f"{D1_string_list}\n")
    file_.write(f"Activation P1: {activation_p1}, P1: {activation_t1}, R1: {activation_r1}, D1: {activation_d1}\n")
    file_.write(f"\nPerturbations method: {('WordNet-coPOS' if copos is True else 'BERT-no coPOS')}.\n")    
    file_.write(f"WordNet selection via: {('majority vote (mode)' if wordnet_mode is True else 'minority vote (least frequent)')}.\n")    
    file_.write(f"Perturbations scenario: {perturbation_scenario}.\n")
    file_.write(f"Perturbations budget: {perturbation_budget}, budget per-sentence: {bps}\n")
    file_.write(f"\tMax L_{lp_norm}-norm: {lp_robustness_perdim}\n")
    file_.write(f"\tMin cosine similarity: {cosine_robustness}\n")    
    file_.write("P1: Structural Probe.\n")
    file_.write(f"\tSDR: {sm}, UUAS: {uuas}, Spearman: {sr}\n")
    file_.write(f"\tSDR': {sm_prime}, UUAS': {uuas_prime}, Spearman': {sr_prime}\n")
    file_.write(f"\tΔ(SDR): {sm-sm_prime}, ΔUUAS: {uuas-uuas_prime}, Δ(Spearman): {sr-sr_prime}\n")
    file_.write("T1: POS-tag.\n")
    file_.write(f"\tAccuracy: {accuracy_t1} ({(accuracy_t1)*max_length:.3f} words correctly guessed)\n")
    file_.write(f"\tAccuracy': {accuracy_prime_t1} ({(accuracy_prime_t1)*max_length:.3f} words correctly guessed)\n")
    file_.write(f"\tΔ(Accuracy): {accuracy_t1-accuracy_prime_t1} ({(accuracy_t1-accuracy_prime_t1)*max_length:.3f} drop of words correctly guessed, {(accuracy_t1-accuracy_prime_t1)*max_length/perturbation_budget:.3f} drop of words correctly guessed per unit of budget)\n")
    file_.write("R1: Root identification.\n")
    file_.write(f"\tAccuracy: {accuracy_r1}\n")
    file_.write(f"\tAccuracy': {accuracy_prime_r1}\n")
    file_.write(f"\tΔ(Accuracy): {accuracy_r1-accuracy_prime_r1}\n")
    file_.write("D1: Depth of the tree.\n")
    file_.write(f"\tSDR: {sd_ratio_d1}, Spearman: {sp_ratio_d1}\n")
    file_.write(f"\tSDR': {same_distance_prime_d1}, Spearman': {sp_ratio_prime_d1}\n")#
    file_.write(f"\tΔ(SDR): {sd_ratio_d1-same_distance_prime_d1}, Δ(Spearman): {sp_ratio_d1-sp_ratio_prime_d1}\n")
    file_.write('\n\n')
    file_.write('\n\n')