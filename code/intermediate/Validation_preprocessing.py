#!/usr/bin/env python
# coding: utf-8

# ## Preprocessing the validation dataset.

# - First step: split the punctation with word in the tokens, and correct grammar and spelling.
# - Second step: fix the tokens like "I'm" to "I" and "'m".
# - Final step: add 'is_negative' feature.

# ### Import

# In[3]:


import os
import pandas as pd
import nltk
from nltk import word_tokenize, pos_tag, RegexpParser
from nltk.tree import Tree
import re, string
#!python3 -m pip install --user pyspellchecker
from spellchecker import SpellChecker
#!pip install autocorrect
from autocorrect import Speller
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


# ### Split the punctuations from the word in tokens and grammar correction.

# This step will split the punctuations from the word in each token, create new tokens for the punctuations and assign them the tag 'O'. Meanwhile, it uses a spell correcter to fix the spelling of the words.

# In[5]:


spell = Speller(lang='en')

# read validation data file and write to another file.
with open('data/ner_tagged.tsv', 'r') as outf, open("data/split_punc_val.tsv", 'w') as inf:
    for line in outf:
        line = line.strip()
        if 'DOCSTART' in line:
          inf.write("-DOCSTART-\t-X-\tO\t\n")
          continue
        if line == "":
            inf.write("\t\t\t\n")
            continue
        word, x, _, tag = line.split("\t")
        if word.replace(".", "", 1).isdigit():
          # if . appears in a digit.
            inf.write(word+"\t"+x+"\t"+_+"\t"+tag+"\n")
            continue
        parts = []
        for match in re.finditer(r'[^.,?!\s]+|[.,?!]', word):
             parts.append(match.group())
        for p in parts:
            if p in string.punctuation:
            # if p is a punctuation.
                tag = "O"
                inf.write(p+"\t"+x+"\t"+_+"\t"+tag+"\n")
            else:
            # if p is a word, then correct the spell and write to new file.
                p = spell(p)
                inf.write(p+"\t"+x+"\t"+_+"\t"+tag+"\n")


# ### re-tokenize the sentences in the validation data set

# This step will re-tokenizes the validation data set, fix the issues like "I'm" to "i" and "'m".

# In[9]:


sents = []

with open("data/split_punc_val.tsv") as f, open("data/validation_set_tokens_fixed.tsv", 'w') as result:
  sent = ""
  for line in f:
    token = []
    line = line.strip()
    if 'DOCSTART' in line:
          result.write("-DOCSTART-\t-X-\tO\t\n")
          continue
    if line == "":
      sents.append(sent.strip())
      sent = ""
      result.write("\t\t\t\n")
      continue
    word, x, _, tag = line.split("\t")
    word = word.lower()
    sent = sent + " " + word

    token = word_tokenize(word)
    initial = 0
    for tok in token:
      if tag.startswith("O"):
        result.write(tok+"\t"+x+"\t"+_+"\t"+tag+"\n")
      elif tag.startswith("I-"):
        result.write(tok+"\t"+x+"\t"+_+"\t"+tag+"\n")
      elif tag.startswith("B"):
        if initial == 0:
          result.write(tok+"\t"+x+"\t"+_+"\t"+tag+"\n")
          initial += 1
        else:
          if tag.startswith("B-N"):
            tag = tag.split("-")[2]
            result.write(tok+"\t"+x+"\t"+_+"\t""I-N-"+tag+"\n")
          else:
            tag = tag.split("-")[1]
            result.write(tok+"\t"+x+"\t"+_+"\t""I-"+tag+"\n")

# delete the middle processing file
os.remove("data/split_punc_val.tsv")


# It will produce some tokens like '' from 60'', which is used to stand for inches.

# In[10]:


word_tokenize('60"')


# In[11]:


len(sents)


# In[12]:


print(sents[0])


# ### Add "is_negative" feature.

# In[13]:


data = pd.read_csv('data/validation_set_tokens_fixed.tsv', sep='\t')


# In[14]:


data.rename(columns = {'Unnamed: 3':'Tags','-DOCSTART-':'Tokens','-X-':'X', 'O':'_'}, inplace = True)


# In[15]:


data


# In[16]:


for i, row in data.reset_index().iterrows():
    if type(row.Tokens) == str:
        if row.Tokens.startswith("un") and "-N-" in row.Tags and row.Tokens != "undermounted":
            data.at[i,'Tags'] = row.Tags.replace("N-", "")
    if type(row.Tokens) == str:
        if row.Tokens.endswith("less") or row.Tokens.endswith("less.") and row.Tokens not in ["screw", "less"] and "-N-" in row.Tags:
            data.at[i,'Tags'] = row.Tags.replace("N-", "")


# In[17]:


data


# In[18]:


import numpy as np
is_negative = []
for i, row in data.reset_index().iterrows():
    if type(row.Tokens) == str:
        if "-N-" in row.Tags:
            is_negative.append(True)
            data.at[i, "Tags"] = row.Tags.replace("N-", "")
        else:
            is_negative.append(False)
    else:
        is_negative.append(np.NaN)


# In[19]:


data


# In[20]:


len(is_negative)


# In[21]:


data["is_negative"] = is_negative


# In[22]:


data.to_csv("data/final_reannotated_val.csv")


# In[23]:


data

