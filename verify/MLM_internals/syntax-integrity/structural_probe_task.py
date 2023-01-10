"""
Alternative to structural probe with convolution-1D model
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
parser.add_argument("-m", "--mlm", dest="mlm", type=str, default='bert',
                    help="Masked Language Model (`glove`, `glove-counterfitted`, `bert`, `roberta`, `bert-finetuned`, `word2vec`, `fasttext` TODO:{`gpt-2`}).")
parser.add_argument("-sa", "--specify-accuracy", dest="specify_mlm_accuracy", type=str, default='',
                    help="Specify the accuracy (in the filename) of a fine-tuned Masked Language Model (this variable is substantially ignored if empty and should not be used on any model but `bert-finetuned`).")
parser.add_argument("-ms", "--mlm-size", dest="mlm_size", type=str, default='base',
                    help="Type of Masked Language Model (`base`, `large`) (ignored if Word2Vec is used).")
parser.add_argument("-d", "--dataset", dest="dataset", type=str, default='ted',
                    help="Space separated datasets from {`ted`, `ud-english-lines`, `ud-english-pud`, `en-universal`}, e.g., `ted ud-english-lines`.")
parser.add_argument("-c", "--classification", dest="classification", type=str, default="True",
                    help="Whether to perform classification or regression on the P1 task.")
parser.add_argument("-mt", "--max-texts", dest="max_texts", type=int, default=-1,
                    help="Number of texts to use in the analysis.")
parser.add_argument("-ml", "--max-length", dest="max_length", type=int, default=20,
                    help="Number of words per-text to use in the analysis (padded/cut if too long/short).")
parser.add_argument("-mlm-bs", "--mlm-batch-size", dest="mlm_batch_size", type=int, default=512,
                    help="Number of samples processed in batch by an mlm (ignored if `mlm` is different from an available mlm). Reduce it on machines with reduced memory.")
parser.add_argument("-r", "--random-sampling", dest="random_sampling", type=str, default="True",
                    help="Whether or not to randomly shuffle the inputs of a dataset (otherwise they are taken sequentially, from the beginning).")
parser.add_argument("-n", "--num-layers", dest="num_layers", type=int, default=4,
                    help="Number of hidden dense layers (>1).")
parser.add_argument("-t", "--train-split", dest="train_split", type=float, default=0.9,
                    help="Percentage of data reserved to train (the remaining part is used to test).")
parser.add_argument("-e", "--epochs", dest="epochs", type=int, default=250,
                    help="Training epochs.")  
parser.add_argument("-s", "--seed", dest="seed", type=int, default=42,
                    help="Seed.")
parser.add_argument("-b", "--baseline", dest="baseline", type=str, default="False",
                    help="The input X (i.e., the embeddings H) is substituted with a random sample (this can be used to have the performances on random noise).")
parser.add_argument("-adj", "--adjacency", dest="task_adjacency", type=str, default="False",
                    help="When set to true, the model learns to re-construct from H->A, where A is the adjacency matrix (0, 1 values) distilled from M.")
parser.add_argument("-pt", "--plot-trees", dest="plot_trees", type=str, default="False",
                    help="Plot and save the predicted trees (10 for training, 10 for test).")

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
task_adjacency = (True if args.task_adjacency.lower()=="true" else False)
plot_trees = (True if args.plot_trees.lower()=="true" else False)

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
save_prefix = f"./../../../data/datasets/conll/{dataset}-{mlm}/"
os.makedirs(save_prefix, exist_ok=True)  # create the folder in case it doesn't exists
S_file = Path(save_prefix+f"S_maxlen-{max_length}{'' if mlm!='bert-finetuned' else f'_acc-{specify_mlm_accuracy}'}.npy")
M_file = Path(save_prefix+f"M_maxlen-{max_length}{'' if mlm!='bert-finetuned' else f'_acc-{specify_mlm_accuracy}'}.npy")
if S_file.is_file() and M_file.is_file():
    S = np.load(str(S_file), allow_pickle=True)
    M = np.load(str(M_file), allow_pickle=True)
else:
    S, M = get_trees_from_conlldatasets(dataset, maxlen=max_length, nsamples=-1)
    M[M==999] = 0.
    # Shuffle data
    if random_sampling is True:
        print("[WARNING] data is now being shuffled: be sure that you are using S,M and H that are correctly `aligned`.")
        M_shape = M.shape
        z = list(zip(S, M))
        random.shuffle(z)
        S, M = zip(*z)
        S, M = list(S), np.array(M)
        assert M.shape == M_shape
    # Save
    np.save(str(S_file), S)
    np.save(str(M_file), M)

# Eventually apply the M->A transformation (distance matrix to adjacency)
if task_adjacency is True:
    M[M!=1.] = 0


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
H_pathname = save_prefix+f"H_maxlen-{max_length}{suffix_layer}{'' if mlm!='bert-finetuned' else f'_acc-{specify_mlm_accuracy}'}.npy"
H_file = Path(H_pathname)
if H_file.is_file():
    X = np.load(H_file._str, allow_pickle=True)
    embedding_dim =  X.shape[1]
    trailing_dims = X.shape[1:]
else:
    X = []
    if mlm in ['word2vec', 'glove', 'glove-counterfitted', 'fasttext']:
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
    model.add(Dense(1500, input_shape=input_shape, activation=activation))
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
    model.add(Dense(1500, activation=activation))
# output layer
if classification is True:
    model.add(Dense(np.prod(output_shape), activation=custom_softmax))
    model.add(Reshape(output_shape))
    model.compile(optimizer='adam', metrics=['mae', 'binary_crossentropy'], loss='binary_crossentropy')
else:
    model.add(Dense(output_shape, activation='linear'))
    model.compile(optimizer='adam', metrics=['mae', 'mse'], loss=('mse' if task_adjacency is False else 'mae'))

# Train
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
h = model.fit(X_train, M_train,
        batch_size=512,
        epochs=epochs,
        verbose=1,
        validation_data=(X_test, M_test),
        callbacks=[callback]
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

string_list = []
model.summary(line_length=80, print_fn=lambda x: string_list.append(x))
string_list = '\n'.join(string_list)
adjacency_suffix = ('' if task_adjacency is False else '_TASK-ADJACENCY')
baseline_suffix = ('' if baseline is False else '_BASELINE-random-inputs')
finetuned_accuracy_suffix = ('' if baseline is False else '_BASELINE-random-inputs')
with open(f'./results/structural/{dataset}_{mlm}{adjacency_suffix}{baseline_suffix}.txt', 'a+') as file_:
    file_.write(f"{'#'*100}\n")
    file_.write(f"Seed: {seed}\n")
    file_.write(f"model: {architecture}, embeddings: {mlm} (layer {layer})\n")
    # Substantially equivalent to if mlm == `bert-finetuned`
    if len(specify_mlm_accuracy) > 0:
        file_.write(f"Fine-tuned accuracy {specify_mlm_accuracy}\n")
    file_.write(f"dataset(s): {dataset}, input-length: {max_length}, classification: {classification}\n")
    file_.write(f"{string_list}\n")
    file_.write(f"Activations: {activation}\n")
    file_.write(f"Same distance ratio: {sm}\n")
    file_.write(f"UUAS: {uuas}\n")
    file_.write(f"Spearman ratio pairwise (mean, std): {sr_pairwise}\n")
    file_.write(f"Spearman ratio: {sr}\n")
    file_.write(f"Spearman ratio (adjacency): {sr_adj}\n")
    file_.write(f"loss: {p[0]} - mae: {p[1]} - mse (binary-crossentropy): {p[2]}")
    file_.write('\n\n')

if plot_trees is True:
    # Train
    indices_to_plot = np.random.permutation(n_train)[:10]
    trees_save_prefix_train = f'./results/structural/trees/{dataset}_{mlm}{adjacency_suffix}{baseline_suffix}/train/layer_{layer}/'
    os.makedirs(trees_save_prefix_train+'M/', exist_ok=True)
    os.makedirs(trees_save_prefix_train+'M_hat/', exist_ok=True)
    for idx in indices_to_plot:
        s = S[idx].split(' ')
        m = M[idx].reshape(1, max_length, max_length)
        if classification is False:
            m_hat = np.round(model.predict(X_train[idx].reshape(1, *X_train.shape[1:])).reshape(1, max_length, max_length))
        else:
            m_hat = np.argmax(model.predict(X_train[idx].reshape(1, *X_train.shape[1:])), axis=2).reshape(1, max_length, max_length)
        draw_graph(s, m, trees_save_prefix_train+'M/'+f'[{idx}].png')
        draw_graph(s, m_hat, trees_save_prefix_train+'M_hat/'+f'[{idx}].png')

    # Test
    indices_to_plot = np.random.permutation(len(X_train))[:10]
    indices_to_plot += len(X_test)  # test offset
    trees_save_prefix_test = f'./results/structural/trees/{dataset}_{mlm}{adjacency_suffix}{baseline_suffix}/test/layer_{layer}/'
    os.makedirs(trees_save_prefix_test+'M/', exist_ok=True)
    os.makedirs(trees_save_prefix_test+'M_hat/', exist_ok=True)
    for idx in indices_to_plot:
        s = S[idx].split(' ')
        m = M[idx].reshape(1, max_length, max_length)
        if classification is False: 
            m_hat = np.round(model.predict(X[idx].reshape(1, *X_test.shape[1:])).reshape(1, max_length, max_length))
        else:
            m_hat = np.argmax(model.predict(X[idx].reshape(1, *X_test.shape[1:])), axis=2).reshape(1, max_length, max_length)
        draw_graph(s, m, trees_save_prefix_test+'M/'+f'[{idx}].png')
        draw_graph(s, m_hat, trees_save_prefix_test+'M_hat/'+f'[{idx}].png')