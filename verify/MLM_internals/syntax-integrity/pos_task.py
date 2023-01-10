"""
POS-tagging.
RQ. Is syntax representation transferrable?
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

import matplotlib.pyplot as plt
from keras.utils import to_categorical
from pathlib import Path
from sklearn.preprocessing import normalize as scipy_normalize
from tensorflow.keras.activations import softmax
from tensorflow.keras.layers import Dense, Flatten, Input, Conv1D, Conv2D, Bidirectional, LSTM, concatenate, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.random import set_seed
from transformers import AdamW, BertTokenizer, BertModel, BertConfig, BertForMaskedLM
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig

from t1_metrics import accuracy
sys.path.append('./../')
from data import get_pos_from_conlldatasets, draw_graph
sys.path.append('./../../../train')
from glove_utils import load_embedding
from BERTModels import SentimentalBERT

def custom_softmax(t):
    global POS
    import tensorflow.keras as K
    n_classes = int(np.max(POS)+1)
    sh = t.shape
    partial_sm = []
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
parser.add_argument("-m", "--mlm", dest="mlm", type=str, default='bert',
                    help="Masked Language Model (`glove`, `glove-counterfitted`, `bert`, `roberta`, `bert-finetuned`, `word2vec`, `fastext` TODO:{`gpt-2`}).")
parser.add_argument("-sa", "--specify-accuracy", dest="specify_mlm_accuracy", type=str, default='',
                    help="Specify the accuracy (in the filename) of a fine-tuned Masked Language Model (this variable is substantially ignored if empty and should not be used on any model but `bert-finetuned`).")
parser.add_argument("-ms", "--mlm-size", dest="mlm_size", type=str, default='base',
                    help="Type of Masked Language Model (`base`, `large`) (ignored if Word2Vec is used).")
parser.add_argument("-d", "--dataset", dest="dataset", type=str, default='ted',
                    help="Space separated datasets from {`ted`, `ud-english-lines`, `ud-english-pud`, `en-universal`}, e.g., `ted ud-english-lines`.")
parser.add_argument("-c", "--classification", dest="classification", type=str, default="True",
                    help="Whether to perform classification or regression on the POS-tagging task.")
parser.add_argument("-mt", "--max-texts", dest="max_texts", type=int, default=-1,
                    help="Number of texts to use in the analysis.")
parser.add_argument("-ml", "--max-length", dest="max_length", type=int, default=20,
                    help="Number of words per-text to use in the analysis (padded/cut if too long/short).")
parser.add_argument("-mlm-bs", "--mlm-batch-size", dest="mlm_batch_size", type=int, default=512,
                    help="Number of samples processed in batch by an mlm (ignored if `mlm` is different from an available mlm). Reduce it on machines with reduced memory.")
parser.add_argument("-r", "--random-sampling", dest="random_sampling", type=str, default="False",
                    help="Whether or not to randomly shuffle the inputs of a dataset (otherwise they are taken sequentially, from the beginning).")
parser.add_argument("-n", "--num-layers", dest="num_layers", type=int, default=4,
                    help="Number of hidden dense layers (>1).")
parser.add_argument("-t", "--train-split", dest="train_split", type=float, default=0.9,
                    help="Percentage of data reserved to train (the remaining part is used to test).")
parser.add_argument("-e", "--epochs", dest="epochs", type=int, default=250,
                    help="Training epochs.")  
parser.add_argument("-s", "--seed", dest="seed", type=int, default=-1,
                    help="Seed.")
parser.add_argument("-b", "--baseline", dest="baseline", type=str, default="False",
                    help="The input X (i.e., the embeddings H) is substituted with a random sample (this can be used to have the performances on random noise).")
parser.add_argument("-k", "--keep-ratio", dest="keep_pos_ratio", type=float, default=0.2,
                    help="Percentage of POS tags kept.")

# global variables
args = parser.parse_args()
architecture = str(args.architecture)
activation = str(args.activation)
layer = int(args.layer)
mlm = str(args.mlm).lower()
specify_mlm_accuracy = str(args.specify_mlm_accuracy)  # accuracy of bert fine-tuned model
mlm_size = str(args.mlm_size)
dataset = list(str(args.dataset).lower().split(' '))
classification = (True if args.classification.lower()=="true" else False)
max_texts = int(args.max_texts)
max_length = int(args.max_length)
mlm_batch_size = int(args.mlm_batch_size)
random_sampling = (True if args.random_sampling.lower()=="true" else False)
num_layers = int(args.num_layers)
train_split = float(args.train_split)
epochs = int(args.epochs)
seed = int(args.seed)
baseline = (True if args.baseline.lower()=="true" else False)
keep_pos_ratio = float(args.keep_pos_ratio)

# Warning on mlm_batch_size
if mlm_batch_size > 1:
    print(f"[WARNING] mlm_batch_size is equal to {mlm_batch_size}, reduce this number if you run out of memory (minimum is 1, but makes the computations slow with mlms!).")

if mlm == 'bert-finetuned':
    assert len(specify_mlm_accuracy) > 0
else:
    assert len(specify_mlm_accuracy) == 0

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

# Sorting prevents from saving permutations of the same datasets
dataset.sort()  

# Get the sentences and the distance matrices (possibly from pre-saved files)
print(f"Collecting texts and POS tags from `{dataset}` dataset...")
save_prefix = f"./../../../data/datasets/conll/{dataset}-{mlm}/probing/"
os.makedirs(save_prefix, exist_ok=True)  # create the folder in case it doesn't exists
S_file = Path(save_prefix+f"S-POS_maxlen-{max_length}_keep-POS-ratio-{keep_pos_ratio}{'' if mlm!='bert-finetuned' else f'_acc-{specify_mlm_accuracy}'}.npy")
POS_file = Path(save_prefix+f"POS_maxlen-{max_length}_keep-POS-ratio-{keep_pos_ratio}{'' if mlm!='bert-finetuned' else f'_acc-{specify_mlm_accuracy}'}.npy")
if S_file.is_file() and POS_file.is_file():
    S = np.load(str(S_file), allow_pickle=True)
    POS = np.load(str(POS_file), allow_pickle=True)
else:
    S, S_POS, POS_FREQ, POS = get_pos_from_conlldatasets(dataset, nsamples=-1, filter_ratio=keep_pos_ratio)
    # Cut max length
    for i,s,p in zip(range(len(S)), S, POS):
        s_split = S[i].split(' ')
        S[i] = ' '.join(s_split[:max_length])
        POS[i] = POS[i][:max_length]
        # pad the POS
        POS[i] += [0]*(max_length-len(POS[i]))
    POS = np.array(POS)
    # Shuffle data
    if random_sampling is True:
        print("[WARNING] data is now being shuffled: be sure that you are using S,P and H that are correctly `aligned`.")
        z = list(zip(S, POS))
        random.shuffle(z)
        S, POS = zip(*z)
        S, POS = list(S), np.array(M)
        assert len(S) == len(POS)
    # Save
    np.save(str(S_file), S)
    np.save(str(POS_file), POS)

# Extract the representations
print(f"Warming-up {mlm}...")
if mlm in ['bert', 'bert-finetuned']:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=True)
    if mlm == 'bert-finetuned':
        bert = BertForMaskedLM.from_pretrained('bert-base-uncased', config=config)
    else:
        bert = BertModel.from_pretrained('bert-base-uncased', config=config)
    # mlm mask token id
    mask_id = tokenizer.encode('[MASK]', add_special_tokens=False)[0]
    cls_id = tokenizer.encode('[CLS]', add_special_tokens=False)[0]
    sep_id = tokenizer.encode('[SEP]', add_special_tokens=False)[0]
    pad_id = tokenizer.encode('[PAD]', add_special_tokens=False)[0]
    pad_token = '[PAD]'
    # Eventually load the pre-trained parameters
    if mlm == 'bert-finetuned':
        bert_weights_path = f'./../../../data/models/bert-finetuned/bert_pretrained_sst_saved_weights_inputlen-20_accuracy-{specify_mlm_accuracy}.pt'
        print(f"Loading pre-saved weights at `{bert_weights_path}`")
        bert = SentimentalBERT(bert, max_length, mlm=True)
        bert.load_state_dict(torch.load(bert_weights_path))
    # Freeze the parameters
    for param in bert.parameters():
        param.requires_grad = False

elif mlm == 'roberta':
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    config = RobertaConfig.from_pretrained("roberta-base", output_hidden_states=True, output_attention=True)
    roberta = RobertaModel.from_pretrained('roberta-base', config=config)
    # mlm mask token id
    mask_id = tokenizer.encode('<mask>', add_special_tokens=False)[0]
    cls_id = tokenizer.encode('<s>', add_special_tokens=False)[0]
    sep_id = tokenizer.encode('</s>', add_special_tokens=False)[0]
    pad_id = tokenizer.encode('<pad>', add_special_tokens=False)[0]
    pad_token = '<pad>'
    # Freeze the parameters
    for param in roberta.parameters():
        param.requires_grad = False

elif mlm == 'glove':
    embedding_filename = './../../../data/embeddings/glove.840B.300d.txt'
    word2toindex, index2word, index2embedding = load_embedding(embedding_filename, try_first='./../../../data/embeddings/')

elif mlm == 'glove-counterfitted':
    embedding_filename = './../../../data/embeddings/glove-counterfitted.840B.300d.txt'
    word2toindex, index2word, index2embedding = load_embedding(embedding_filename)

elif mlm == 'word2vec':
    embedding_filename = './../../../data/embeddings/google.news.300d.txt'
    word2toindex, index2word, index2embedding = load_embedding(embedding_filename)

elif mlm == 'fasttext':
    embedding_filename = './../../../data/embeddings/fasttext.en.300d.txt'
    word2toindex, index2word, index2embedding = load_embedding(embedding_filename)

elif mlm == 'gpt-2':
    pass
    
else:
    raise Exception(f"{mlm} is not a valid value for mlm.")

# Collect the input representations
print(f"Collecting the {mlm} texts representations...")
suffix_layer = (f'_layer-{[layer]}' if mlm in ['bert', 'roberta', 'bert-finetuned'] else '')
H_pathname = save_prefix+f"H_maxlen-{max_length}{suffix_layer}_keep-POS-ratio-{keep_pos_ratio}{'' if mlm!='bert-finetuned' else f'_acc-{specify_mlm_accuracy}'}.npy"
H_file = Path(H_pathname)
if H_file.is_file():
    X = np.load(H_file._str, allow_pickle=True)
    embedding_dim =  X.shape[1]
    trailing_dims = X.shape[1:]
else:
    X = []
    if mlm in ['word2vec', 'glove', 'glove-counterfitted', 'word2vec', 'fasttext']:
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
        for batch_range in tqdm.tqdm(range(0, len(S), mlm_batch_size)):
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
            if mlm == 'bert':
                x_p1 = bert(BERT_INPUTS, attention_mask=INPUT_MASKS)[2][layer][:,:,:] # collect all the tokens, then average
            elif mlm == 'bert-finetuned':
                x_p1 = bert(BERT_INPUTS, mask=INPUT_MASKS)[1][layer][:,:,:] # collect all the tokens, then average
            elif mlm == 'roberta':
                x_p1 = roberta(BERT_INPUTS, attention_mask=INPUT_MASKS)[2][layer][:,:,:]
            else: # gpt
                raise Exception(f"{mlm} is not a valid masked language model.")
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
n_pos = int(np.max(POS))+1
POS = POS.reshape(len(POS), max_length)
if classification is True:
    POS = to_categorical(POS)
n_train, n_test = int(train_split*len(X)), int((1.-train_split)*len(X))
X_train, POS_train = X[:n_train], POS[:n_train]
X_test, POS_test = X[n_train:], POS[n_train:]
assert len(X_train) == len(POS_train) and len(X_test) == len(POS_test)

# Define the custom model shapes
input_shape = X[0].shape
distances_shape = POS[0].shape
output_shape = (max_length,n_pos)

# Create the model
model = Sequential()
if architecture == 'fc':
    model.add(Dense(512, input_shape=input_shape, activation=activation))
elif architecture == 'cnn':
    model.add(Conv1D(256, kernel_size=embedding_dim, strides=embedding_dim, input_shape=input_shape, activation=activation))
elif architecture == 'cnn2d':
    model.add(Conv2D(256, kernel_size=(embedding_dim, 2), strides=(1, 1), input_shape=input_shape, activation=activation))
    model.add(Flatten())
elif architecture == 'rnn':
    model.add(LSTM(256, return_sequences=True, input_shape=input_shape))
    model.add(Flatten())
    #model.add(Bidirectional(LSTM(256)))
else:
    raise Exception(f"{architecture} is not a valid architecture")

# Add the deeper layers
for _ in range(num_layers-1):
    model.add(Dense(512, activation=activation))
# output layer
if classification is True:
    model.add(Dense(np.prod(output_shape), activation=custom_softmax))
    model.add(Reshape(output_shape))
    model.compile(optimizer='adam', metrics=['mae', 'mse', 'binary_crossentropy'], loss='binary_crossentropy')
else:
    model.add(Dense(max_length, activation='linear'))
    model.compile(optimizer='adam', metrics=['mae', 'mse', 'categorical_crossentropy'], loss='mse')

# Train
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
h = model.fit(X_train, POS_train,
        batch_size=512,
        epochs=epochs,
        verbose=1,
        validation_data=(X_test, POS_test),
        callbacks=[callback]
        )

# Test
print(f"\nTest set evaluation:")
p = model.evaluate(X_test, POS_test)

if architecture == 'fc':
    POS_hat = model.predict(X_test.reshape(len(X_test), np.prod(trailing_dims)))
elif architecture == 'cnn':
    POS_hat = model.predict(X_test.reshape(len(X_test), embedding_dim, max_length)).squeeze(1)
elif architecture == 'cnn2d':
    POS_hat = model.predict(X_test.reshape(len(X_test), embedding_dim, max_length, 1)).squeeze(1)
elif architecture == 'rnn':
    POS_hat = model.predict(X_test.reshape(len(X_test), embedding_dim, max_length))
else:
    raise Exception(f"{architecture} is not a valid architecture")

# Check some predictions on test
print("WORD | POS | POS_HAT")
for s_test, p_test, p_hat in zip(S[n_train:n_train+10], POS_test[:10], POS_hat[:10]):
    s_test_split = s_test.split(' ')
    for w,tag,tag_hat in zip(s_test_split, p_test, p_hat):
        if classification is True:
            print(f"`{w}` | {np.argmax(tag)} | {np.argmax(tag_hat)}")
        else:
            print(f"`{w}` | {tag} | {int(np.round(tag_hat))}")
    print("\n")

a = accuracy(POS_test, POS_hat, classification=classification)
print(f"\nAccuracy: {a}")

string_list = []
model.summary(line_length=80, print_fn=lambda x: string_list.append(x))
string_list = '\n'.join(string_list)
baseline_suffix = ('' if baseline is False else '_BASELINE-random-inputs')
os.makedirs('./results/probing_task/', exist_ok=True)
with open(f'./results/probing_task/{dataset}_{mlm}{baseline_suffix}.txt', 'a+') as file_:
    file_.write(f"{'#'*100}\n")
    file_.write(f"Seed: {seed}\n")
    file_.write(f"model: {architecture}, embeddings: {mlm} (layer {layer})\n")
    # Substantially equivalent to if mlm == `bert-finetuned`
    if len(specify_mlm_accuracy) > 0:
        file_.write(f"Fine-tuned accuracy {specify_mlm_accuracy}\n")
    file_.write(f"dataset(s): {dataset}, keep_pos_ratio: {keep_pos_ratio}, input-length: {max_length}, classification: {str(classification)}\n")
    file_.write(f"{string_list}\n")
    file_.write(f"Activations: {activation}\n")
    file_.write(f"Accuracy: {a}\n")
    file_.write(f"loss: {p[0]} - mae: {p[1]} - mse: {p[2]} - cross_entropy: {p[3]}")
    file_.write('\n\n')
