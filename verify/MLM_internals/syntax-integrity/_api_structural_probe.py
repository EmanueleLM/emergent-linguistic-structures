"""
api to run a structural probe on an external model (so far, BERT-base based)
"""
import argparse
import copy as cp
import itertools
import linecache
import logging
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

from keras.utils import to_categorical
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import normalize as scipy_normalize
from tensorflow.keras.activations import softmax
from tensorflow.keras.layers import Dense, Flatten, Input, Conv1D, Conv2D, Bidirectional, LSTM, concatenate, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.random import set_seed
from transformers import AdamW, BertTokenizer, BertModel, BertConfig, BertForMaskedLM
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig

from p1_metrics import UUAS, same_distance, spearman_pairwise, spearman
sys.path.append('./../')
from data import get_trees_from_conlldatasets, draw_graph
sys.path.append('./../../../train')
from glove_utils import load_embedding
from BERTModels import SentimentalBERT

n_pos = None  # this will be updated as the algorthm evolves

def custom_softmax(t):
    global n_pos
    import tensorflow.keras as K
    n_classes = n_pos  # the diameter of the line-graph tree
    sh = t.shape
    partial_sm = []
    #print(sh[1] // n_classes)
    for i in range(sh[1] // n_classes):
        partial_sm.append(softmax(t[:, i*n_classes:(i+1)*n_classes]))
    return concatenate(partial_sm)

def evaluate_syntactic_performances(representation, architecture='fc', activation='relu', layer=-9, mlm='bert', mlm_size='base',
                                    dataset='ted', classification=True, max_texts=-1, max_length=20, mlm_batch_size=512, 
                                    random_sampling=True, num_layers=2, train_split=0.9, epochs=250, baseline=False, task_adjacency=False
                                    ):
    
    global n_pos
    # Warning on mlm_batch_size
    if mlm_batch_size > 1:
        print(f"[WARNING] mlm_batch_size is equal to {mlm_batch_size}, reduce this number if you run out of memory (minimum is 1, but makes the computations slow with mlms!).")

    # Log settings
    logger = logging.getLogger(__name__)

    # Adjacency?
    if task_adjacency is True:
        print("\n[WARNING] Adjacency mode is activated, matrices M are replaced with adjacencies (0., 1. values).\n")

    # Baseline?
    if baseline is True:
        print("\n[WARNING] Baseline mode is activated, X (i.e., the embeddings H) are replaced with random noise!\n")

    # Sorting prevents from saving permutations of the same datasets
    dataset.sort()  

    # Get the sentences and the distance matrices (possibly from pre-saved files)
    print(f"Collecting texts and distance matrices from `{dataset}` dataset...")
    S, M = get_trees_from_conlldatasets(dataset, maxlen=max_length, nsamples=-1)
    M[M==999] = 0.
    # Eventually apply the M->A transformation (distance matrix to adjacency)
    if task_adjacency is True:
        M[M!=1.] = 0

    # Extract the representations
    print(f"Warming-up {mlm}...")
    if mlm in ['bert', 'bert-finetuned']:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=True)
        bert = representation  # make the api compliant with the old code
        # mlm mask token id
        mask_id = tokenizer.encode('[MASK]', add_special_tokens=False)[0]
        cls_id = tokenizer.encode('[CLS]', add_special_tokens=False)[0]
        sep_id = tokenizer.encode('[SEP]', add_special_tokens=False)[0]
        pad_id = tokenizer.encode('[PAD]', add_special_tokens=False)[0]
        pad_token = '[PAD]'
        # Freeze the parameters
        for param in bert.parameters():
            param.requires_grad = False  
    else:
        raise Exception(f"{mlm} is not a valid value for mlm.")

    # Collect the input representations
    print(f"Collecting the {mlm} texts representations...")
    X = []
    # Create bert inputs and masks
    for batch_range in tqdm.tqdm(range(0, len(S[:1000]), mlm_batch_size)):
        BERT_INPUTS, INPUT_MASKS = [], []
        for s in tqdm.tqdm(S[batch_range:batch_range+mlm_batch_size]):
            s_split = s.split(' ')
            s_pad = s_split + [pad_token]*(max_length-len(s_split))
            bert_input = torch.tensor(tokenizer.convert_tokens_to_ids(s_pad)).reshape(1,-1)
            input_mask = torch.tensor([1 for _ in s_split]+[0 for _ in range(max_length-len(s_split))]).reshape(1,-1)
            BERT_INPUTS += [bert_input]
            INPUT_MASKS += [input_mask]
        BERT_INPUTS = torch.tensor([b.numpy() for b in BERT_INPUTS]).reshape(len(BERT_INPUTS), max_length)
        INPUT_MASKS = torch.tensor([i.numpy() for i in INPUT_MASKS]).reshape(len(INPUT_MASKS), max_length)
        if mlm in ['bert', 'bert-finetuned']:
            x_p1 = bert.get_reps(BERT_INPUTS, mask=INPUT_MASKS)[1][layer][:,:,:] # collect all the tokens, then average
        else: # gpt
            raise Exception(f"{mlm} is not a valid masked language model.")
        X += [[x] for x in x_p1]
    embedding_dim =  x_p1.shape[2]
    trailing_dims = x_p1.shape[1:]
    X = np.array([x[0].numpy() for x in X]).reshape(len(X), embedding_dim, max_length)

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
    if classification is True:
        n_pos = int(np.max(M)+1)  # number of classes in the classification setting
        M = to_categorical(M)
        M = M.reshape(len(M), max_length*max_length, n_pos)
    else:
        M = M.reshape(len(M), max_length*max_length)
    n_train, n_test = int(train_split*len(X)), int((1.-train_split)*len(X))
    X_train, M_train = X[:n_train], M[:n_train]
    X_test, M_test = X[n_train:], M[n_train:]
    assert len(X_train) == len(M_train) and len(X_test) == len(M_test)

    # Create the custom model to allow the 
    input_shape = X[0].shape
    distances_shape = M[0].shape
    output_shape = (max_length*max_length if classification is False else (max_length*max_length, n_pos))

    # Create the model
    model = Sequential()
    if architecture == 'fc':
        model.add(Dense(512, input_shape=input_shape, activation=activation))
        model.add(Dropout(0.3))
    elif architecture == 'cnn':
        model.add(Conv1D(256, kernel_size=embedding_dim, strides=embedding_dim, input_shape=input_shape, activation=activation))
    elif architecture == 'cnn2d':
        model.add(Conv2D(256, kernel_size=(embedding_dim, 2), strides=(1, 1), input_shape=input_shape, activation=activation))
        model.add(Flatten())
    elif architecture == 'rnn':
        model.add(Bidirectional(LSTM(256, return_sequences=True), input_shape=input_shape))
        #model.add(Flatten())
        model.add(Bidirectional(LSTM(256)))
    else:
        raise Exception(f"{architecture} is not a valid architecture")

    # Add the deeper layers
    for _ in range(num_layers-1):
        model.add(Dense(512, activation=activation))
        model.add(Dropout(0.3))
    # output layer
    if classification is True:
        model.add(Dense(np.prod(output_shape), activation=custom_softmax))
        model.add(Reshape(output_shape))
        model.compile(optimizer='adam', metrics=['mae', 'binary_crossentropy'], loss='binary_crossentropy')
    else:
        model.add(Dense(output_shape, activation='relu'))
        model.compile(optimizer='adam', metrics=['mae', 'mse'], loss=('mse' if task_adjacency is False else 'mae'))

    # Train
    callback_earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4, restore_best_weights=True)
    callback_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
    h = model.fit(X_train, M_train,
            batch_size=512,
            epochs=epochs,
            verbose=1,
            validation_data=(X_test, M_test),
            callbacks=[callback_earlystop, callback_scheduler]
            )

    # Test
    print(f"\nTest set evaluation:")
    p = model.evaluate(X_test, M_test)

    if architecture == 'fc':
        M_hat = model.predict(X_test.reshape(len(X_test), np.prod(trailing_dims)))
    elif architecture == 'cnn':
        M_hat = model.predict(X_test.reshape(len(X_test), embedding_dim, max_length)).squeeze(1)
    elif architecture == 'cnn2d':
        M_hat = model.predict(X_test.reshape(len(X_test), embedding_dim, max_length, 1))
    elif architecture == 'rnn':
        M_hat = model.predict(X_test.reshape(len(X_test), embedding_dim, max_length))
    else:
        raise Exception(f"{architecture} is not a valid architecture")

    if classification is True:
        M = np.argmax(M, axis=2).reshape(len(X), max_length, max_length)
        M_hat = np.argmax(M_hat, axis=2).reshape(len(X_test), max_length, max_length)
        M_test = np.argmax(M_test, axis=2).reshape(len(X_test), max_length, max_length)

    sm = same_distance(M_test, M_hat, max_length)
    sr_pairwise = spearman_pairwise(M_test, M_hat, max_length)
    sr = spearman(M_test, M_hat, max_length)
    sr_adj = spearman(M_test, M_hat, max_length, adjacency=True)
    uuas = UUAS(M_test, M_hat, max_length)  

    return uuas, sm, sr, X_test[:100]

