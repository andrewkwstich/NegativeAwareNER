#!/usr/bin/env python
# coding: utf-8

# In[76]:


import pandas as pd


# In[77]:


data = pd.read_csv("./un-negated_clean_data.csv")


# In[78]:


data


# In[79]:


data = data[["Tokens","Tags","sentence"]]


# In[80]:


data


# In[81]:


data['Sentence']= data[['sentence','Tokens','Tags']].groupby(['sentence'])['Tokens'].transform(lambda x: ' '.join(x))


# In[82]:


data['Tags']= data[['sentence','Tokens','Tags']].groupby(['sentence'])['Tags'].transform(lambda x: ','.join(x))


# In[83]:


data = data[['Sentence','Tags']]


# In[84]:


data = data.drop_duplicates().reset_index(drop=True)


# In[85]:


data


# In[86]:


train_size = 0.8
train_dataset = data.sample(frac=train_size,random_state=200)
test_dataset = data.drop(train_dataset.index).reset_index(drop=True)


# In[87]:


def strsplit_tags(tags):
    sent_list = tags.split(",")
    return sent_list


# In[88]:


def strsplit_sentence(sentence):
    sent_list = sentence.split(" ")
    return sent_list


# In[89]:


train_sents = []

for index, row in train_dataset.iterrows():
    train_sents.append((strsplit_sentence(row['Sentence']),strsplit_tags(row['Tags'])))


# In[90]:


dev_sents = []

for index, row in test_dataset.iterrows():
    dev_sents.append((strsplit_sentence(row['Sentence']),strsplit_tags(row['Tags'])))


# In[91]:



import nltk


def get_pos(word):
    tag = nltk.pos_tag([word])
    return tag[0][1]

def is_number(string):
    return any(char.isdigit() for char in string)

def word2features(sentence, idx):
    word_features = {}
    word_features['word_lowercase'] = sentence[idx].lower()

    # your code here
    ## Features looking at the neighbouring words:
    
    if idx > 0:
        word_features["pre_word"] = sentence[idx -1].lower()
    else:
        word_features["pre_word"] = ""
    if idx < len(sentence) - 1:
        word_features["next_word"] = sentence[idx +1].lower()
    else:
        word_features["next_word"] = ""
        
    if idx > 1:
        word_features["pre2_word"] = sentence[idx -2].lower()
    else:
        word_features["pre2_word"] = ""
        
    if idx < len(sentence) - 2:
        word_features["next2_word"] = sentence[idx +2].lower()
    else:
        word_features["next2_word"] = ""
    ## Features loking at the word endings
    
    if len(sentence[idx])> 2:
        word_features["last2char"] = sentence[idx][-2:]
    else:
        word_features["last2char"] = sentence[idx]
    
    if len(sentence[idx])> 3:
        word_features["last3char"] = sentence[idx][-3:]
    else:
        word_features["last3char"] = sentence[idx]
        
    ## Features considering the shape of the word
    
    if sentence[idx].isupper():
        word_features["upper"] = True
    else:
        word_features["upper"] = False  
        
    if sentence[idx].islower():
        word_features["lower"] = True
    else:
        word_features["lower"] = False 
    
    word_features["length"] = len(sentence[idx])
    word_features["position"] = idx
    
    ## Gazetteer features:
    

    
    ## Extra Features:
    
    ## Is Number
    word_features["number"] = is_number(sentence[idx])
    
#     if is_number(sentence[idx]) == True:
#         word_features["num_length"] = len(sentence[idx])
#     else:
#         word_features["num_length"] = 0
    
    ##Is_Noun
    
    if get_pos(sentence[idx])== "NN":
        word_features["is_noun"] = True
    else:
        word_features["is_noun"] = False
    
    ## Percent Feature:
    
        
    for i in range(len(sentence)):
        if i == idx and i != 0:
            word_features['first_word_not_in_title_case'] = sentence[idx].istitle()
        elif i == idx and i == 0:
            if sentence[idx].istitle():
                word_features['first_word_not_in_title_case'] = False
    
        
    return word_features
    
    
def sentence2features(sentence):
    return [word2features(sentence, idx) for idx in range(len(sentence))]


# In[92]:


def prepare_ner_feature_dicts(sents):
    '''ner_files is a list of Ontonotes files with NER annotations. Returns feature dictionaries and 
    IOB tags for each token in the entire dataset'''
    all_dicts = []
    all_tags = []
    # your code here
    for tokens, tags in sents:
        all_dicts.extend(sentence2features(tokens))
        all_tags.extend(tags)
    
    return all_dicts, all_tags


# In[93]:


train_dicts, train_tags = prepare_ner_feature_dicts(train_sents)
dev_dicts, dev_tags = prepare_ner_feature_dicts(dev_sents)


# In[94]:


len(train_dicts)


# In[95]:


len(train_tags)


# In[96]:



import os
import nltk
nltk.download("gazetteers")
nltk.download("names")
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import f1_score, classification_report
from bs4 import BeautifulSoup
from nltk.corpus import names,gazetteers 
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_f1_score, flat_classification_report
import re


# In[97]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import DictVectorizer

vec = DictVectorizer()
train_dicts = vec.fit_transform(train_dicts)
dev_dicts = vec.transform(dev_dicts)
clf = MultinomialNB()
clf.fit(train_dicts, train_tags)
y_pred = clf.predict(dev_dicts)

# your code here
print("MicroF1:",f1_score(dev_tags, y_pred,average="micro"))
print("MacroF1:",f1_score(dev_tags, y_pred,average="macro"))
print(classification_report(dev_tags, y_pred))


# ### CRF

# In[98]:


def prepare_ner_feature_dicts(sents):
    '''ner_files is a list of Ontonotes files with NER annotations. Returns feature dictionaries and 
    IOB tags for each token in the entire dataset'''
    all_dicts = []
    all_tags = []
    # your code here
    for tokens, tags in sents:
        all_dicts.append(sentence2features(tokens))
        all_tags.append(tags)

    return all_dicts, all_tags


# In[99]:


train_dicts, train_tags = prepare_ner_feature_dicts(train_sents)
dev_dicts, dev_tags = prepare_ner_feature_dicts(dev_sents)


# In[100]:


import sklearn_crfsuite

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=350,
    all_possible_transitions=True,
    verbose = True
)
crf.fit(train_dicts, train_tags)
try:
    call_produces_an_error()
except:
    pass


# In[101]:


def flatten(l):
    result = []
    for sub in l:
        result.extend(sub)
    return result

y_pred = crf.predict(dev_dicts)
print(f1_score(flatten(dev_tags), flatten(y_pred), average='macro'))
print(f1_score(flatten(dev_tags), flatten(y_pred), average='micro'))
print(classification_report(flatten(dev_tags), flatten(y_pred)))


# In[ ]:




