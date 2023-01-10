"""
Tree-root task classification.
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

from r1_metrics import accuracy as r1_accuracy
sys.path.append('./../')
from data import get_roots_from_conlldatasets
sys.path.append('./../../../train')
from glove_utils import load_embedding
from BERTModels import SentimentalBERT

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

# Parse command line args
parser = argparse.ArgumentParser()
parser.add_argument("-a", "--architecture", dest="architecture", type=str, default='fc',
                    help="Architecture of the model that learns the H->C mapping (`fc`, `cnn`, `rnn`, `cnn2d`).")
parser.add_argument("-act", "--activation", dest="activation", type=str, default='relu',
                    help="Activation of each network layer (`linear`, `tanh`, `relu`, etc.).")
parser.add_argument("-l", "--layer", dest="layer", type=int, default=-2,
                    help="Layer from which representations are extracted (ignored if Word2Vec is used).")
parser.add_argument("-m", "--llm", dest="llm", type=str, default='bert',
                    help="Masked Language Model (`glove`, `glove-counterfitted`, `bert`, `roberta`, `bert-finetuned`, `word2vec`, `fasttext` TODO:{`gpt-2`}).")
parser.add_argument("-sa", "--specify-accuracy", dest="specify_llm_accuracy", type=str, default='',
                    help="Specify the accuracy (in the filename) of a fine-tuned Masked Language Model (this variable is substantially ignored if empty and should not be used on any model but `bert-finetuned`).")
parser.add_argument("-ms", "--llm-size", dest="llm_size", type=str, default='base',
                    help="Type of Masked Language Model (`base`, `large`) (ignored if Word2Vec is used).")
parser.add_argument("-d", "--dataset", dest="dataset", type=str, default='ted',
                    help="Space separated datasets from {`ted`, `ud-english-lines`, `ud-english-pud`, `en-universal`}, e.g., `ted ud-english-lines`.")
parser.add_argument("-mt", "--max-texts", dest="max_texts", type=int, default=-1,
                    help="Number of texts to use in the analysis.")
parser.add_argument("-ml", "--max-length", dest="max_length", type=int, default=20,
                    help="Number of words per-text to use in the analysis (padded/cut if too long/short).")
parser.add_argument("-llm-bs", "--llm-batch-size", dest="llm_batch_size", type=int, default=512,
                    help="Number of samples processed in batch by an llm (ignored if `llm` is different from an available llm). Reduce it on machines with reduced memory.")
parser.add_argument("-r", "--random-sampling", dest="random_sampling", type=str, default="True",
                    help="Whether or not to randomly shuffle the inputs of a dataset (otherwise they are taken sequentially, from the beginning).")
parser.add_argument("-n", "--num-layers", dest="num_layers", type=int, default=1,
                    help="Number of hidden dense layers (>1).")
parser.add_argument("-t", "--train-split", dest="train_split", type=float, default=0.9,
                    help="Percentage of data reserved to train (the remaining part is used to test).")
parser.add_argument("-e", "--epochs", dest="epochs", type=int, default=250,
                    help="Training epochs.")  
parser.add_argument("-s", "--seed", dest="seed", type=int, default=42,
                    help="Seed.")
parser.add_argument("-b", "--baseline", dest="baseline", type=str, default="False",
                    help="The input X (i.e., the embeddings H) is substituted with a random sample (this can be used to have the performances on random noise).")

# global variables
args = parser.parse_args()
architecture = str(args.architecture)
activation = str(args.activation)
layer = int(args.layer)
llm = str(args.llm).lower()
specify_llm_accuracy = str(args.specify_llm_accuracy)  # accuracy of bert fine-tuned model
llm_size = str(args.llm_size)
dataset = list(str(args.dataset).lower().split(' '))
max_texts = int(args.max_texts)
max_length = int(args.max_length)
llm_batch_size = int(args.llm_batch_size)
random_sampling = (True if args.random_sampling.lower()=="true" else False)
num_layers = int(args.num_layers)
train_split = float(args.train_split)
epochs = int(args.epochs)
seed = int(args.seed)
baseline = (True if args.baseline.lower()=="true" else False)

# Warning on llm_batch_size
if llm_batch_size > 1:
    print(f"[WARNING] llm_batch_size is equal to {llm_batch_size}, reduce this number if you run out of memory (minimum is 1, but makes the computations slow with llms!).")

if llm == 'bert-finetuned':
    assert len(specify_llm_accuracy) > 0
else:
    assert len(specify_llm_accuracy) == 0

# Log settings
logger = logging.getLogger(__name__)

# Set seed
if seed == -1:
    seed = np.random.randint(0, 10000)
print(f"Seed of the session: `{seed}`.")
os.environ['PYTHONHASHSEED']=str(seed)
random.seed(seed)
np.random.seed(seed)
set_seed(seed)

# Baseline?
if baseline is True:
    print("\n[WARNING] Baseline mode is activated, X (i.e., the embeddings H) are replaced with random noise!\n")

# Write the header of the results
os.makedirs('./results/robustness/tree_root_task/', exist_ok=True)
with open(f'./results/robustness/tree_root_task/results.txt', 'a+') as file_:
    file_.write(f"{'#'*30} BEGIN {'#'*30}\n")
    file_.write(f"\nSeed: {seed}\n")
    file_.write(f"\nDataset: {dataset}, baseline: {baseline}\n")

# Sorting prevents from saving permutations of the same datasets (look at argument --dataset)
dataset.sort()  

# Get the sentences and the distance matrices (possibly from pre-saved files)
print(f"Collecting texts and distance matrices from `{dataset}` dataset...")
save_prefix = f"./../../../data/datasets/conll/{dataset}-{llm}/"
os.makedirs(save_prefix, exist_ok=True)  # create the folder in case it doesn't exists
R_file = Path(save_prefix+f"R_maxlen-{max_length}{'' if llm!='bert-finetuned' else f'_acc-{specify_llm_accuracy}'}.npy")
S_file = Path(save_prefix+f"S_maxlen-{max_length}{'' if llm!='bert-finetuned' else f'_acc-{specify_llm_accuracy}'}.npy")
if R_file.is_file():
    print(f"Loading files from {R_file} and {S_file} paths.")
    R = np.load(str(R_file), allow_pickle=True)
    S = np.load(str(S_file), allow_pickle=True)
else:
    S, R = get_roots_from_conlldatasets(dataset, maxlen=max_length, nsamples=-1)  # please notice that S is unused but kept for compatibility
    if random_sampling is True:
        print("[warning] data is now being shuffled: be sure that you are using S,R and H that are correctly `aligned`.")
        R_shape = R.shape
        z = list(zip(S, R))
        random.shuffle(z)
        S, R = zip(*z)
        S = list(S)
        R = np.array(R)
        assert len(R) == len(S)
        if np.sum(R) != len(R):
            print(f"[warning] Some of the sentences have been the root set to the minimum value of their connection, not to the real root (value of connection 0.)")
    # Save
    print(f"Saving files to {R_file} and {S_file} paths.")
    np.save(str(R_file), R)
    #np.save(str(S_file), S)

# Extract the representations
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
    # Eventually load the pre-trained parameters
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

# Collect the input representations
print(f"Collecting the {llm} texts representations...")
suffix_layer = (f'_layer-{[layer]}' if llm in ['bert', 'roberta', 'bert-finetuned'] else '')
H_pathname = save_prefix+f"H_maxlen-{max_length}{suffix_layer}{'' if llm!='bert-finetuned' else f'_acc-{specify_llm_accuracy}'}.npy"
H_file = Path(H_pathname)
if H_file.is_file():
    X = np.load(H_file._str, allow_pickle=True)
    embedding_dim =  X.shape[1]
    trailing_dims = X.shape[1:]
else:
    raise Exception(f"[ERROR] {H_pathname} does not exist. Root task requires structural probe to be collected beforehand.")

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

# Create the custom model to allow the 
input_shape = X[0].shape
distances_shape = R[0].shape
output_shape = max_length

# Create the model
model = Sequential()
if architecture == 'fc':
    model.add(Dense(2048, input_shape=input_shape, activation=activation))
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
    model.add(Dense(2048, activation=activation))
# output layer
model.add(Dense(output_shape, activation="softmax"))
model.compile(optimizer='adam', metrics=['mae', 'categorical_crossentropy'], loss='categorical_crossentropy')

# Train
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
h = model.fit(X_train, R_train,
        batch_size=512,
        epochs=epochs,
        verbose=1,
        validation_data=(X_test, R_test),
        callbacks=[callback]
        )

# Test
print(f"\nTest set evaluation:")
p = model.evaluate(X_test, R_test)

if architecture == 'fc':
    R_hat = model.predict(X_test.reshape(len(X_test), np.prod(trailing_dims)))
elif architecture == 'cnn':
    R_hat = model.predict(X_test.reshape(len(X_test), embedding_dim, max_length)).squeeze(1)
elif architecture == 'cnn2d':
    R_hat = model.predict(X_test.reshape(len(X_test), embedding_dim, max_length, 1))
elif architecture == 'rnn':
    R_hat = model.predict(X_test.reshape(len(X_test), embedding_dim, max_length))
else:
    raise Exception(f"{architecture} is not a valid architecture")

accuracy = r1_accuracy(R_test, R_hat)

print(f"Dataset(s): {dataset}")
print(f"Accuracy over {len(R_test)} samples: {accuracy}")

with open(f'./results/robustness/tree_root_task/results.txt', 'a+') as file_:
    file_.write(f"Accuracy: {accuracy}\n")
    file_.write(f"{'#'*30} END {'#'*30}\n")
