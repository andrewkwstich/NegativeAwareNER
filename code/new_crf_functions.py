#!/usr/bin/env python
# coding: utf-8

import spacy
import pandas as pd
import ast
import numpy as np
from wordfreq import word_frequency
from gensim.models.keyedvectors import KeyedVectors
nlp = spacy.load("en_core_web_sm")

#TODO Use different word embeddings
import gensim.downloader as api
glove_vectors=api.load("glove-wiki-gigaword-300")

from scipy.spatial import distance


with open('data/dpt_store_centroids.txt') as f:
    dpt_store_centroids = f.read()
        
# reconstructing the data as a dictionary
list_taxon_dict = ast.literal_eval(dpt_store_centroids)

taxon_dict = {}

for k, v in list_taxon_dict.items():
    taxon_dict[k] = np.array(v)
    
def cosine_similarity(first_word, second_word):
    return 1 - distance.cosine(first_word, second_word)

def get_sim_freq(sent, i, taxon_dict, verbose=False):
    """
    Returns the ratio between the similarity between the word and the taxon to which it is most similar and the word frequency.
    This is to account for the fact that more frequent words tend to be assigned higher similarity scores to other words
    even when not intuitively semantically similar.
    The ratio is divided by 10 for readability.
    Then, onverts similarity-frequency ratio to an ordinal representation by representing it as a set of binary variables.
    This is because CRFsuite cannot use numerical features directly.
    """
    
    highest_similarity = 0
    sent = nlp(" ".join([str(tok) for tok in sent]))
    word = sent[i]
    if word.lemma_ in glove_vectors:
        if word.pos_ in {"PROPN", "NOUN", "ADJ"}:
            for k, v in taxon_dict.items():
                similarity = cosine_similarity(glove_vectors[word.lemma_], v)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    taxon = k
                if verbose:
                    print(k, ":", similarity)
                    
    elif word.pos_ == "PROPN":
        highest_similarity = 1
    
    if verbose:
        print("\nTaxon with highest similarity :", taxon)

    if not highest_similarity:
        highest_similarity = 0
        
    frequency = word_frequency(word.lemma_, "en") * 1000
    
    if frequency != 0 and highest_similarity > 0 and highest_similarity != 1:
        sim_freq = highest_similarity / frequency / 10
    
    elif highest_similarity == 1:
        sim_freq = 1
        
    else:
        sim_freq = 0
    
    zero = sim_freq == 0
    one = sim_freq == 1
    tenth_half = sim_freq >= 0.1 and sim_freq != 1
    half_one = sim_freq >= 0.5 and sim_freq != 1
    one_three = sim_freq > 1
    three_five = sim_freq >= 3
    five_ten = sim_freq >= 5
    greater_than_ten = sim_freq >= 10
    
    return zero, one, tenth_half, half_one, one_three, three_five, five_ten, greater_than_ten

def is_object(sent, i, n=0):
    """
    Returns whether the word at index i is an object, object modifier, or neither.
    These categories are more semantic than syntactic. For example, in a sentence such as "Does your X have Y attribute?",
    X and Y will be considered as object and object modifier, respectively, rather than subject and object.
    """
    
    sent = nlp(" ".join([str(tok) for tok in sent]))
    
    attribute_list = ["size", "color", "colour", "material", "texture", "shape"]
    
    if i >= len(sent):
        return "not_object"
    
    if n >= 10:
        return "not_object"
    
    if sent[i].text in attribute_list:
        return "object_modifier"
    
    if sent[i].dep_ == "pobj" and sent[i].head.text == "in" and sent[sent[i].head.i-1].lemma_ == "come":
        return "object_modifier"
    
    elif sent[i].dep_ == "ROOT" and sent[i].pos_ == "NOUN":
        return "object"
    
    elif sent[i].dep_ in ["pobj", "dobj"] and sent[i].pos_ != "PRON":
        if sent[i].head.dep_ in ["ROOT", "xcomp", "ccomp"] or (sent[i].head.dep_ == "conj" and sent[i].head.head.dep_ in ["ROOT", "xcomp", "ccomp"]):
            if sent[i].head.dep_ in ["ROOT", "xcomp", "ccomp"]:
                head_lemma = sent[i].head.lemma_
            elif sent[i].head.dep_ == "conj" and sent[i].head.head.dep_ in ["ROOT", "xcomp", "ccomp"]:
                head_lemma = sent[i].head.head.lemma_
            if head_lemma != "have":
                return "object"
            else:
                for word in sent:
                    if word.head.i == sent[i].head.i:
                        if word.text == "i":
                            return "not_object"
                        elif "subj" in word.dep_ and word.pos_ == "NOUN":
                            return "object_modifier"
                    elif word.head.text == "of" and word.head.head.text == "any" and  word.head.head.head.i == sent[i].head.i:
                        return "object_modifier"

            if word.head.pos_ == "ADP" and sent[word.head.i-1].dep_ in ["pobj", "dobj"]:
                return "object_modifier"

            return "object"
        
        elif sent[i].head.pos_ == "ADP" and sent[i].head.head.dep_ in ["ROOT", "xcomp", "ccomp"] and sent[i].head.head.lemma_ == "look":
            if sent[i].pos_ != "ADJ":
                return "object"
            else:
                return "object_modifier"
        
        elif is_object(sent, sent[i].head.i-1, n+1) in ["object", "object_modifier"]:
            return "object_modifier"
        
        elif sent[i].head.text == "of" and sent[i].head.head.text == "any" and sent[i].head.head.dep_ == "nsubj":
            return "object"
        
        elif sent[i].head.text == "than" and sent[i].head.head.dep_ == "acomp":
            if is_object(sent, sent[i].head.head.head.i, n+1) == "object":
                return "object"
            elif is_object(sent, sent[i].head.head.head.head.i, n+1) == "object":
                return "object"
        
        else:
            return "not_object"
        
    elif sent[i].dep_ == "punct":
        if sent[i].head.dep_ == "nummod":
            return "object_modifier"
        elif sent[i].text == "-":
            if is_object(sent, sent[i].head.i, n+1) in ["object", "object_modifier"]:
                return "object_modifier"
            else:
                return "not_object"
        else:
            return "not_object"
        
            
    elif "mod" in sent[i].dep_:
        if is_object(sent, i+1, n+1) in ["object", "object_modifier"]:
            return "object_modifier"
        else:
            return "not_object"
        
    elif sent[i].dep_ == "compound":
        if is_object(sent, i+1, n+1) in ["object", "object_modifier"]:
            return "object_modifier"
        else:
            return "not_object"

    elif sent[i].dep_ in ["conj", "appos"] and (sent[i].pos_ != "VERB" or sent[i].tag_ in ["VBG", "VBN"]):
        if is_object(sent, sent[i].head.i, n+1) == "object" and sent[i].pos_ in ["NUM", "ADJ"] and sent[i].head.pos_ == "NOUN":
            return "object_modifier"
        else:
            return is_object(sent, sent[i].head.i, n+1)
    
    elif sent[i].dep_ == "relcl" and (sent[i].pos_ != "VERB" or sent[i].tag_ in ["VBG", "VBN"]):
        if is_object(sent, sent[i].head.i, n+1) in ["object_modifier", "object"]:
            return "object_modifier"
        else:
            return "not_object"
    
    elif "subj" in sent[i].dep_ and sent[i].pos_ == "NOUN" and sent[i].head.lemma_ in ["have", "come", "be"]:
        return "object"
    
    elif sent[i].dep_ == "acomp":
        return "object_modifier"

    else:
        return "not_object"

