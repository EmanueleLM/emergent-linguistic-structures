# pip install -U pytorch-pretrained-bert
import copy as cp
import itertools
import nltk
import numpy as np
import random
import torch

from collections import Counter
from tensorflow.keras.models import load_model
from transformers import BertTokenizer
from nltk.corpus import wordnet as wn

def mode(a, reverse=False):
    """
    Reverse returns the least common element
    """
    c = Counter(a)
    if reverse is False:
        return c.most_common(1)[0][0]
    else:
        return c.most_common(len(c))[-1][0]

import logging
logging.basicConfig(level=logging.INFO)

DEVICE = torch.device('cpu')

# Import WordNet
nltk.download('wordnet')

def random_combination(iterable, r, sims):
    i = 0
    pool = tuple(iterable)
    n = len(pool)
    rng = range(n)
    while i < sims:
        i += 1
        rr = random.randint(1, r)
        yield [pool[j] for j in random.sample(rng, rr)]

def mapIndexValues(a, b):
    """
    Extract elements in a (list) using indices in b (list:int)
    """
    out = map(a.__getitem__, b)
    return list(out)

def tokenize_text(text, return_indices=False, use_bert=True, TOKENIZER=None):
    """
    TODO: complete documentation
    use_bert:boolean
     (optional) use BERT tokenizer to split the input string, otherwise use split function.
    TOKENIZER:bert-tokenizer
      (optional) In your code, the tokenizer should be set as a valid tokenizer, e.g.:
      from transformers import BertTokenizer;TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
        or
     from transformers import AutoTokenizer;TOKENIZER = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')"
    """
    if use_bert is True:
        tokenized_text = TOKENIZER.tokenize(text)
    else:
        tokenized_text = text.split(' ')
    if return_indices is True:
        return TOKENIZER.convert_tokens_to_ids(tokenized_text)
    return tokenized_text

def targeted_interventions(model, masked_text, original_text, topk=5, mask='[MASK]', predicts_only=[], words_per_concept=10, budget=100, device='cpu', TOKENIZER=None):
    """
    Predicts a subset of masked tokens in a text, generating subsitutions from the first token on left to the last on the right.
    model:function 
     returns a prediction on an input text
    masked_text:list
     tokenized text (tokens in a concept are masked)
    original_text:list
     tokenized text (tokens in a concept are not masked): this is used when not all the words in a concepts are masked
    masked_text:list
     tokenized text (with mask tokens)
    topk:int
     (optional) number of replacements returned per-word
    predicts_only:list
     (optional) list of lists of indices from the masked_text on which BERT predicts the substitutions 
      Substitutions in lists inside the main list are considered independent from each other.
      All the other `masks` are ignored. If predicts_only==[], it predicts on every `masked` tokens.
    words_per_concept:int
     (optional) limit the number of words that are evaluated in each concept (i.e., each list)
      (5 is a good measure, greater values melt the machine).
    budget:int
     (optional) maximum number of permutations returned at each iteration
    TOKENIZER:bert-tokenizer
      (optional) In your code, the tokenizer should be set as a valid tokenizer, e.g.:
      from transformers import BertTokenizer;TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
        or
     from transformers import AutoTokenizer;TOKENIZER = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')"
    """
    global DEVICE
    flatten_indices_predicts_only, flatten_indices_nomask = [], []
    for po in predicts_only:
        flatten_indices_predicts_only += po[:words_per_concept]
        flatten_indices_nomask += po
    masked_indices = [i for i,w in enumerate(masked_text) if w == mask and (i in flatten_indices_predicts_only if predicts_only!=[] else True)]
    # If there is nothing to predict, return
    if len(masked_indices) == 0:
        return {}
    # Restore the original text when not predicting a masked token
    for i,_ in enumerate(original_text):
        if i not in flatten_indices_predicts_only and i in flatten_indices_nomask:
            masked_text[i] = original_text[i]
    # Re-generate masked indices but now in compsact lists per-concept
    masked_indices = [p[:words_per_concept] for p in predicts_only]  # alternative name: `indices_predicts_only`
    # Tokenize text
    masked_texts = [[[TOKENIZER.convert_tokens_to_ids(masked_text)]] for _ in masked_indices]
    masked_texts_prob = [[[1.]] for _ in masked_indices]
    # Start genereating perturbations
    predicted_tokens = {k:[] for k in flatten_indices_predicts_only}
    for scp_index, single_concept_interventions in enumerate(masked_indices):  # for each concept at the same `hierarchy` level
        for midx in single_concept_interventions:  # for each concept
            tmp, tmp_prob = [], []
            #print(f"Processing index {midx} out of {single_concept_interventions}")
            #print(f"{len(masked_texts[scp_index][-1][:budget])} will be processed")
            for mt,mt_prob in zip(masked_texts[scp_index][-1][:budget], masked_texts_prob[scp_index][-1][:budget]):  # for each masked word
                segments_ids = [0] * len(mt)
                tokens_tensor = torch.tensor([mt]).to(DEVICE)
                segments_tensors = torch.tensor([segments_ids]).to(DEVICE)
                with torch.no_grad():
                    predictions = model(tokens_tensor, segments_tensors)[0]
                    #return predictions
                    #print(predictions.shape)
                    predictions = torch.nn.functional.softmax(predictions, dim=2)
                    #raw_prediction = model(tokens_tensor, segments_tensors)[0]
                # random shuffling of the indices
                topk_indices = torch.topk(predictions[0, midx], topk).indices
                #print(torch.topk(predictions[0, midx], topk))
                #print(torch.topk(predictions[0, midx], topk).shape)
                #oo
                topk_probs = torch.topk(predictions[0, midx], topk).values
                #print(topk_probs)
                #print(torch.topk(raw_prediction[0, midx], topk))
                #print()
                #random.shuffle(topk_indices)  # de-comment to shuffle indices
                for r,p in zip(topk_indices, topk_probs):
                    mt_copy = cp.copy(mt)
                    mt_copy[midx] = int(r)
                    tmp += [mt_copy]
                    # store probabilities of each generation
                    mt_prob *= float(p)
                    tmp_prob += [mt_prob]
                    if len(tmp) > budget:
                        break
                if len(tmp) > budget:
                    break
            masked_texts[scp_index] += [tmp]
            masked_texts_prob[scp_index] += [tmp_prob]
            # sort both the lists based on the odds of a word to occur
            masked_texts[scp_index][-1] = [x for _,x in sorted(zip(masked_texts_prob[scp_index][-1],masked_texts[scp_index][-1]), reverse=True)]
            masked_texts_prob[scp_index][-1] = sorted(masked_texts_prob[scp_index][-1], reverse=True)
            #print(masked_texts_prob[scp_index][-1])
        for mt in masked_texts[scp_index][-1][:budget]:
            #print(f"Processing {mt} on indices {masked_indices}")
            #print(predicted_tokens.keys())
            #print((mt, masked_indices[scp_index]))
            MIV = TOKENIZER.convert_ids_to_tokens(mapIndexValues(mt, masked_indices[scp_index]))
            for k, miv in zip(masked_indices[scp_index], MIV):
                predicted_tokens[k] += [miv]
    #print(predicted_tokens)
    return predicted_tokens

def sequential_interventions(model, text, interventions, topk=1, combine_subs=False, mask="[MASK]", budget=250, copos=False, device='gpu', TOKENIZER=None):
    """
    Use this function to perturb tokens in a text following a the order specified in an intervention list.
    Input:
    model:function
        function used to predict the output class of the input text
    text:list
        a list of words, already tokenized (not masked for observational, masked for the first step of interventional)
    interventions:list
        list of lists (of indices) of words that are affected by the interventions. 
        The order of the lists specifies the relationship between parents and children. 
        In the same list each sub-item is a concept list (each are independent)
    topk:int
        (optional) number of replacements generated by BERT for each simulation
    combine_subs:boolean
        (optional) combine topk substitutions obtained for each word through their cartesian product
    mask:str
        (optional) token used to mask words to-be-predicted
    budget:int
     (optional) maximum number of permutations returned at each iteration
    copos:boolean
     (optional) whether to substitute words with coPOS (coherent Part-of-Speech)
    TOKENIZER:bert-tokenizer
      (optional) In your code, the tokenizer should be set as a valid tokenizer, e.g.:
      from transformers import BertTokenizer;TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
        or
     from transformers import AutoTokenizer;TOKENIZER = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')"
    Output:
    A list of substitutions, which in turn are lists of word-level tokens
    TODO: any parent influences all the subs, this code is possibly wrong (e.g., in case of disjoint children)
    """
    subs = [cp.copy(text)]
    flatten_indices_masks = []
    for intervention in interventions:
        flatten_indices_masks += [item for sublist in intervention for item in sublist]
    # mask input
    subs[0] = [(s if i not in flatten_indices_masks else mask) for i,s in enumerate(subs[0])]
    new_subs = []
    for i,target_tokens in enumerate(interventions):
        #print(f"Generating interventions for indices {target_tokens}")
        for sub in subs:
            if target_tokens != []:
                tmp_text = cp.copy(sub)
                #print("Plain intervention list ", interventions)
                #print(f"Will intervene on indices {i},{i+1}")
                dict_subs = targeted_interventions(model, tmp_text, text, topk=topk, mask=mask, predicts_only=interventions[i], budget=budget, device=device, TOKENIZER=TOKENIZER)
                #return dict_subs
                #print(f"Interventions on {interventions[i:i+1][0]}")
                #print(dict_subs)
                dict_subs_keys, dict_subs_values = list(dict_subs.keys()), list(dict_subs.values())
                combinations = (zip(*dict_subs_values) if combine_subs is False else itertools.islice(itertools.product(*dict_subs_values), budget))  # set limit to cartesian product
                for el in combinations:
                    tmp_sub = cp.copy(tmp_text)
                    for ii, idx in enumerate(dict_subs_keys):
                        tmp_sub[idx] = el[ii]  
                    new_subs += [tmp_sub]
                    if len(new_subs) > budget:
                        break
                if len(new_subs) > budget:
                    break
        #print(f"{subs}")
        #print(f"{new_subs}")
        subs = cp.copy(new_subs)
        new_subs = []
    return subs[:budget]

def predictPOS(mlm_model, tokenizer, input_text, max_length, layer, mlm='bert', path='./../../../data/models/pos-tag/tagger_maxlen-20_mlm-bert', device='cpu'):
    """
    Use an mlm and a pre-trained model to pos-tag a sentence
    """
    if mlm != 'bert':
        raise Exception(f"{mlm} is not a valid mlm.")
    # Config
    pad_token = '[PAD]'
    # Load model that predicts POS tags
    pos_tagger = load_model(path)
    # Encode the input
    s_pad = input_text + [pad_token]*(max_length-len(input_text))
    bert_input = torch.tensor(tokenizer.convert_tokens_to_ids(s_pad)).reshape(1,-1)
    input_mask = torch.tensor([1 for _ in input_text]+[0 for _ in range(max_length-len(input_text))]).reshape(1,-1)
    # Get the representation
    x = mlm_model(bert_input, attention_mask=input_mask)[1][layer][:,:,:] # collect all the tokens, then average
    x = x.detach().numpy().reshape(1,-1)
    # POS-tag the input
    y_hat = np.argmax(pos_tagger(x), axis=2)
    return y_hat

def wordnet_perturbations(input_text, idx, topk=10, least_common=False):
    input_text_cp = input_text[idx]
    subs = wn.synsets(input_text[idx])
    if len(subs) > 0:
        pos = [s._pos for s in subs[:topk]]
        mode_pos = mode(pos, least_common)
        for el in subs:
            if mode_pos == el._pos:
                likely_sub = el._name.split('.')[0].lower()
                if likely_sub != input_text_cp.lower():
                    return likely_sub
    return input_text_cp