#!/usr/bin/env python
# coding: utf-8

import spacy
import pandas as pd
import ast
import numpy as np
from wordfreq import word_frequency
from gensim.models.keyedvectors import KeyedVectors
nlp = spacy.load("en_core_web_sm")
import gensim.downloader as api
glove_vectors=api.load("glove-wiki-gigaword-300")



from scipy.spatial import distance



def cosine_similarity(first_word, second_word):
    return 1 - distance.cosine(first_word, second_word)



data = pd.read_csv('data/spacy_tokenized.csv')[["Tokens", "Tags"]]


with open('data/dpt_store_centroids.txt') as f:
    dpt_store_centroids = f.read()
        
# reconstructing the data as a dictionary
list_taxon_dict = ast.literal_eval(dpt_store_centroids)

taxon_dict = {}

for k, v in list_taxon_dict.items():
    taxon_dict[k] = np.array(v)


# In[212]:


def get_sentences(df, with_iob=False):
    
    current_sent = ""
    
    all_sents = []
    
    for i, row in df.reset_index().iterrows():
        if type(row.Tags) == str:
            if not with_iob:
                current_sent = current_sent + row.Tokens + " "
            else:
                current_sent = current_sent + row.Tokens + "-" + row.Tags + " "
        else:
            all_sents.append(current_sent[:-1])
            current_sent = ""
            
    return all_sents


# In[213]:


all_sents = get_sentences(data)
all_sents_iob = get_sentences(data, True)


# In[214]:


# def get_highest_similarity(sent, i, verbose=False):
#     """
#     Returns the cosine similarity score between the word at index i and the composite vector representing the taxon
#     to which the word is most similar.
#     If verbose, also prints the name of the taxon to ensure that the output of the function is sensible.
#     """
    
#     highest_similarity = 0
#     sent = nlp(sent)
#     word = sent[i]
#     if word.lemma_ in glove_vectors:
#         if word.pos_ in {"PROPN", "NOUN", "ADJ"}:
#             for k, v in taxon_dict.items():
#                 similarity = cosine_similarity(glove_vectors[word.lemma_], v)
#                 if similarity > highest_similarity:
#                     highest_similarity = similarity
#                     taxon = k
#                 if verbose:
#                     print(k, ":", similarity)
#     elif word.pos_ == "PROPN":
#         return None
                        
#     if verbose:
#         print("\nTaxon with highest similarity :", taxon)
                
#     return highest_similarity
    


# In[239]:


# def get_frequency(sent, i):
#     """
#     Returns the frequency of the word at index i, multiplied by 1000 for readability.
#     """
    
#     sent = nlp(sent)
#     word = sent[i]
    
#     return word_frequency(word.lemma_, "en") * 1000
    


# In[225]:


def get_sim_freq(sent, i, taxon_dict, verbose=False):
    """
    Returns the ratio between the similarity between the word and the taxon to which it is most similar and the word frequency.
    This is to account for the fact that more frequent words tend to be assigned higher similarity scores to other words
    even when not intuitively semantically similar.
    The ratio is divided by 10 for readability.
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
    
    if frequency != 0 and highest_similarity > 0:
        sim_freq = highest_similarity / frequency / 10
    
    else:
        sim_freq = 0

    """
    Converts similarity-frequency ratio to an ordinal representation by representing it as a set of binary variables.
    This is because CRFsuite cannot use numerical features directly.
    """
    
    
    
    zero = sim_freq == 0
    one = sim_freq == 1
    tenth_half = sim_freq >= 0.1
    half_one = sim_freq >= 0.5
    one_three = sim_freq > 1
    three_five = sim_freq >= 3
    five_ten = sim_freq >= 5
    greater_than_ten = sim_freq >= 10
    
    return zero, one, tenth_half, half_one, one_three, three_five, five_ten, greater_than_ten


# In[226]:


def sim_freq_2_ordinal(sim_freq):
    
    """
    Converts similarity-frequency ratio to an ordinal representation by representing it as a set of binary variables.
    This is because CRFsuite cannot use numerical features directly.
    """
    
    
    
    zero = sim_freq == 0
    one = sim_freq == 1
    tenth_half = sim_freq >= 0.1
    half_one = sim_freq >= 0.5
    one_three = sim_freq > 1
    three_five = sim_freq >= 3
    five_ten = sim_freq >= 5
    greater_than_ten = sim_freq >= 10
    
    return zero, one, tenth_half, half_one, one_three, three_five, five_ten, greater_than_ten


# In[229]:


# for sent in all_sents[:20]:
#     # sent = nlp(sent)
#     for i in range(len(sent)):
#         print(sent[i])
#         print("similarity-frequency ratio", get_sim_freq(sent, i, taxon_dict))
#         print()
#     print("\n\n")


# In[169]:


sent = "looking for pre workout pump addict instead of karbolyn hydrate which one is better?"


# In[130]:


all_sents[2] = "i need a 48 inch glass sliding goof and a shower pan system for 500.99 or less"


# In[140]:


all_sents[5] = "does the linen and cotton duvet cover come in king?"


# In[173]:


for word in nlp(sent):
    print(word, word.dep_, word.pos_, word.head)


# In[161]:


def is_object(sent, i):
    """
    Returns whether the word at index i is an object, object modifier, or neither.
    These categories are more semantic than syntactic. For example, in a sentence such as "Does your X have Y attribute?",
    X and Y will be considered as object and object modifier, respectively, rather than subject and object.
    """
    
    sent = nlp(" ".join([str(tok) for tok in sent]))
    # sent = nlp(sent)
    
    if i >= len(sent):
        return "not_object"
    
    if sent[i].dep_ == "pobj" and sent[i].head.text == "in" and sent[sent[i].head.i-1].lemma_ == "come":
        return "object_modifier"
    
    elif sent[i].dep_ in ["pobj", "dobj"] and sent[i].pos_ != "PRON":
        if sent[i].head.dep_ in ["ROOT", "xcomp", "ccomp"] or sent[i].head.dep_ == "conj" and sent[i].head.head.dep_ in ["ROOT", "xcomp", "ccomp"]:
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
        
        elif is_object(sent, sent[i].head.i-1) in ["object", "object_modifier"]:
            return "object_modifier"
        
        elif sent[i].head.text == "of" and sent[i].head.head.text == "any" and sent[i].head.head.dep_ == "nsubj":
            return "object"
        
        elif sent[i].head.text == "than" and sent[i].head.head.dep_ == "acomp":
            if is_object(sent, sent[i].head.head.head.i) == "object":
                return "object"
            elif is_object(sent, sent[i].head.head.head.head.i) == "object":
                return "object"
        
        else:
            return "not_object"
        
    elif sent[i].dep_ == "punct" and sent[i].head.dep_ == "nummod":
        return "object_modifier"

    elif "mod" in sent[i].dep_:
        if is_object(sent, i+1) in ["object", "object_modifier"]:
            return "object_modifier"
        else:
            return "not_object"
        
    elif sent[i].dep_ == "compound":
        if is_object(sent, i+1) in ["object", "object_modifier"]:
            return "object_modifier"
#             if sent[i].pos_ == "ADJ":
#                 return "object_modifier"
#             elif i > 0 and sent[i-1].text.isnumeric():
#                 return "object_modifier"
#             elif sent[i].head.head.pos_ == "ADP" and is_object(sent, sent[i].head.head.i-1) in ["object", "object_modifier"]:
#                 return "object_modifier"
#             else:
#                 return "object"
        else:
            return "not_object"

    elif sent[i].dep_ in ["conj", "appos"] and (sent[i].pos_ != "VERB" or sent[i].tag_ in ["VBG", "VBN"]):
        return is_object(sent, sent[i].head.i)
    
    elif sent[i].dep_ == "relcl" and (sent[i].pos_ != "VERB" or sent[i].tag_ in ["VBG", "VBN"]):
        if is_object(sent, sent[i].head.i) in ["object_modifier", "object"]:
            return "object_modifier"
        else:
            return "not_object"
    
    elif "subj" in sent[i].dep_ and sent[i].pos_ == "NOUN" and sent[i].head.lemma_ in ["have", "come", "be"]:
        return "object"
    
    elif sent[i].dep_ == "acomp":
        return "object_modifier"

    else:
        return "not_object"


# In[162]:


# for sent in all_sents[:10]:
#     for i in range(len(nlp(sent))):
#         print(nlp(sent)[i], nlp(sent)[i].dep_, nlp(sent)[i].tag_, nlp(sent)[i].pos_, nlp(sent)[i].head, nlp(sent)[i].head.dep_, is_object(sent, i))
# #         print(nlp(sent)[i], is_object(sent, i))
#     print()


# In[ ]:




