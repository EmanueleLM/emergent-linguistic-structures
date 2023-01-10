"""
Assessing whether syntactically similar sentences have similar representation
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
from pathlib import Path
from scipy import spatial
from sklearn.cluster import AgglomerativeClustering
from tensorflow.keras.activations import softmax
from tensorflow.keras.layers import Dense, Flatten, Input, Conv1D, Conv2D, Bidirectional, LSTM, concatenate, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.random import set_seed
from transformers import AdamW, BertTokenizer, BertModel, BertConfig, BertForMaskedLM, BertTokenizerFast
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig

from _api_structural_probe import evaluate_syntactic_performances
from p1_metrics import UUAS
sys.path.append('./../')
from data import get_trees_from_conlldatasets, draw_graph
sys.path.append('./../../../train')
from glove_utils import load_embedding
from text_utils_torch import load_SST, load_IMDB, dataset_to_dataloader
from BERTModels import SentimentalBERT

def syntax_collapse(model, x):

    return 0.

# function to train the model
def train():
    global model, cross_entropy, tau, canary
    model.train()
    total_loss = 0
    total_preds=[]
    total_labels = np.array([])
    # iterate over batches
    for step,batch in enumerate(train_dataloader):
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch
        model.zero_grad()        
        preds = model(sent_id, mask)
        loss = cross_entropy(preds, labels)
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        preds=preds.detach().cpu().numpy()
        total_preds.append(preds)
        total_labels = np.append(total_labels, labels.cpu().detach().numpy(), axis=0)
    avg_loss = total_loss / len(train_dataloader)
    total_preds  = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds, total_labels

def evaluate():
    global model, test_dataloader, cross_entropy, device
    print("\nEvaluating...")
    model.eval()
    total_loss = 0
    total_preds, total_labels = np.array([]), np.array([])
    for step,batch in enumerate(test_dataloader):
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(test_dataloader)))
        batch = [t.to(device) for t in batch]
        sent_id, mask, labels = batch
        with torch.no_grad():
            preds = model(sent_id, mask)
            loss = cross_entropy(preds,labels)
            total_loss = total_loss + loss.item()
            preds = preds.detach().cpu().numpy()
            total_preds = np.append(total_preds, [np.argmax(p) for p in preds], axis=0)
            total_labels = np.append(total_labels, labels.cpu().detach().numpy(), axis=0)
    avg_loss = total_loss / len(test_dataloader) 
    return avg_loss, total_preds, total_labels

# specify cpu/GPU
device = torch.device('cpu')

# Logs only errors
tf.get_logger().setLevel('INFO')

# Generic global params
seed = 1

# Training params
dataset = ["sst", "imdb"]
maxlen, epochs = 20, 50
batch_size = 512
tau = 3e-3

# Canary params
canary_dataset = 'en-universal'
canary = []
# canary_size = 1000  # it means that #canary_size pairs will be used to prevent collapsing of syntax structures
layer = -9

# Set seed
if seed == -1:
    seed = np.random.randint(0, 10000)
print(f"Seed of the session: `{seed}`.")
os.environ['PYTHONHASHSEED']=str(seed)
random.seed(seed)
np.random.seed(seed)
set_seed(seed)

# Load BERT classifier
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained("bert-base-uncased",
                                    output_hidden_states=True)
bert = BertForMaskedLM.from_pretrained('bert-base-uncased', config=config)
for param in bert.parameters():
    param.requires_grad = True
# Add the trainable classification `head`
model = SentimentalBERT(bert, maxlen, mlm=False)

# Load dataset
for data in dataset:
    if data == 'sst':
        (X_train, y_train),  (X_test, y_test) = load_SST(maxlen, path='./../../../data/datasets/SST_2/')
    elif data == 'imdb':
        (X_train, y_train),  (X_test, y_test) = load_IMDB(maxlen)
    elif data == '':
        pass 
    else:
        raise Exception(f"{data} is not a valid dataset.")

    n_samples = len(X_test)

    # Create train/test torch dataloaders
    train_dataloader = dataset_to_dataloader(X_train, y_train, tokenizer, maxlen, batch_size)
    test_dataloader = dataset_to_dataloader(X_test, y_test, tokenizer, maxlen, batch_size)

    # Start training
    model = model.to(device)  # push the model to the training device selected
    optimizer = AdamW(model.parameters(), lr=3e-4)  
    cross_entropy = nn.NLLLoss()

    best_valid_loss = float('inf')
    train_losses, valid_losses = [], []
    train_preds, valid_preds = [], []
    ACCURACY_TRAIN, ACCURACY_VALID = [], []
    UUAS, SDR, SR = [], [], []
    COSINE_SIM, L2 = [], []
    indices_samples = np.random.permutation(n_samples)[:100]

    #for each epoch
    prev_samples = []
    for epoch in range(epochs):     
        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))   
        # Train 
        train_loss, total_preds_train, total_labels_train = train() 
        accuracy_train = np.mean([1 if p==l else 0 for p,l in zip(np.argmax(total_preds_train, axis=1), total_labels_train)]) 
        # Validate  
        valid_loss, total_preds, total_labels = evaluate() 
        accuracy_valid = np.mean([1 if p==l else 0 for p,l in zip(total_preds, total_labels)])
        # Check performance on structural probe
        representation = cp.deepcopy(model)
        uuas, sdr, sr, current_samples = evaluate_syntactic_performances(representation=representation, dataset=[canary_dataset])
        """
        #save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'./../../../data/models/bert-finetuned/bert_pretrained_{dataset}_saved_weights_inputlen-{maxlen}_accuracy-{accuracy:.2f}.pt')
        """
        # Append metrics
        UUAS.append(uuas); SDR.append(sdr); SR.append(sr)
        train_losses.append(train_loss); train_preds.append(total_preds_train)
        valid_losses.append(valid_loss); valid_preds.append(total_preds)
        ACCURACY_TRAIN.append(accuracy_train); ACCURACY_VALID.append(accuracy_valid)

        L2_tmp, COS_tmp = [], []
        for s1, s2 in zip(prev_samples, current_samples):
            L2_tmp.append(np.linalg.norm(s1.flatten()-s2.flatten(), ord=2))
            COS_tmp.append(1-spatial.distance.cosine(s1.flatten(), s2.flatten()))
        L2.append(np.max(L2_tmp))
        COSINE_SIM.append(np.min(COS_tmp))

        print(f'uuas, sdr, sr: {uuas:.3f}, {sdr:.3f}, {sr:.3f}')
        print(f'\nTraining loss, Valid loss: {train_loss:.3f}, {valid_loss:.3f}')
        print(f'Train Accuracy, Valid Accuracy: {accuracy_train:.3f}, {accuracy_valid:.3f}')

    #torch.save(model.state_dict(), f'./../../../data/models/bert-finetuned/overfitted_bert_pretrained_{dataset}_saved_weights_inputlen-{maxlen}_accuracy-{accuracy:.2f}.pt')    

    print(f"UUAS: \n{UUAS}\n")
    print(f"SDR: \n{SDR}\n")
    print(f"Spearman: {SR}")
    print(f"TRAIN LOSSES: {train_losses}")
    print(f"VALID LOSSES: {valid_losses}")
    print(f"TRAIN ACCURACIES: {ACCURACY_TRAIN}")
    print(f"VALID ACCURACIES: {ACCURACY_VALID}")

    def wsmooth(x):
        assert len(x) >= 3
        x_smooth = []
        for i in range(len(x)):
            if i == 0:
                x_succ = x[i+1]
                x_smooth.append((x[i]+x_succ)/2)
            elif i == len(x)-1:
                x_prec = x[i-1]
                x_smooth.append((x[i]+x_prec)/2)
            else:
                x_prec = x[i-1]
                x_succ = x[i+1]
                x_smooth.append((x[i]+x_prec+x_succ)/3)
        return x_smooth    

    import matplotlib.pyplot as plt
    x_axis = [i for i in range(epochs)]
    plt.plot(x_axis, wsmooth(UUAS), '-o', c='orange', label='uuas')
    plt.plot(x_axis, wsmooth(SDR), '-o', c='b', label='sdr')
    #plt.plot(x_axis, SR, '-o', c='black', label='sr')
    # plt.plot(x_axis, wsmooth(train_losses), '-^', c='g', label='tloss')
    # plt.plot(x_axis, wsmooth(valid_losses), '.-', c='r', label='vloss')
    plt.legend(loc='best')
    plt.title(f"{data} - Syntax Metrics")
    plt.savefig(f'./results/{data}-smooth-{canary_dataset}-layer[{layer}]-syntax-collapse-1.png')
    plt.savefig(f'./results/{data}-smooth-{canary_dataset}-layer[{layer}]-syntax-collapse-1.svg')
    plt.cla()

    plt.plot(x_axis, wsmooth(train_losses), '-o', c='orange', label='uuas')
    plt.plot(x_axis, wsmooth(valid_losses), '-o', c='b', label='sdr')
    #plt.plot(x_axis, SR, '-o', c='black', label='sr')
    plt.plot(x_axis, wsmooth(ACCURACY_TRAIN), '-^', c='g', label='atrain')
    plt.plot(x_axis, wsmooth(ACCURACY_VALID), '.-', c='r', label='avalid')
    plt.legend(loc='best')
    plt.title(f"{data} - Accuracy and Loss")
    plt.savefig(f'./results/{data}-smooth-{canary_dataset}-layer[{layer}]-syntax-collapse-2.png')
    plt.savefig(f'./results/{data}-smooth-{canary_dataset}-layer[{layer}]-syntax-collapse-2.svg')
    plt.cla()

    plt.plot(x_axis, wsmooth(COSINE_SIM), '-o', c='orange', label='cosine')
    plt.plot(x_axis, wsmooth(L2), '-o', c='b', label='L2')
    #plt.plot(x_axis, SR, '-o', c='black', label='sr')
    # plt.plot(x_axis, wsmooth(ACCURACY_TRAIN), '-^', c='g', label='tacc')
    # plt.plot(x_axis, wsmooth(ACCURACY_VALID), '.-', c='r', label='vacc')
    plt.legend(loc='best')
    plt.title(f"{data} - Norm and Cosine")
    plt.savefig(f'./results/{data}-smooth-{canary_dataset}-layer[{layer}]-syntax-collapse-3.png')
    plt.savefig(f'./results/{data}-smooth-{canary_dataset}-layer[{layer}]-syntax-collapse-3.svg')
    plt.cla()