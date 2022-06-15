#!/usr/bin/env python
# coding: utf-8

# In[5]:


import re, string
import os
import pandas as pd
import nltk
from nltk import word_tokenize, pos_tag, RegexpParser
from nltk.tree import Tree
import re, string
get_ipython().system('python3 -m pip install --user pyspellchecker')
from spellchecker import SpellChecker
get_ipython().system('pip install autocorrect')
from autocorrect import Speller
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


# In[2]:


get_ipython().system('python3 -m pip install --user pyspellchecker')
from spellchecker import SpellChecker

spell2 = SpellChecker()


# In[6]:


import pandas as pd
spell = Speller(lang='en')

# read validation data file and write to another file.
with open('data/ner_tagged.tsv', 'r') as outf, open("data/cleaned_ner_tagged.tsv", 'w') as inf:
    for line in outf:
        line = line.strip()
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


# In[7]:


data = pd.read_csv('data/cleaned_ner_tagged.tsv', sep='\t')
data.rename(columns = {'O':'Tags','-DOCSTART-':'Tokens','-X-':'X'}, inplace = True)


# In[8]:


data


# In[ ]:





# In[9]:


import csv
#Create attributes file from https://nijianmo.github.io/amazon/index.html
known_attributes = set()
with open('data/attributes.csv', newline='\n') as csvfile:
    attributes = csv.reader(csvfile, delimiter=' ')
    for row in attributes:
        for attr in row:
            known_attributes.add(attr.lower())


# In[10]:


neg_sents = []
current_sent = []
current_tags = []
contains_neg = False
non_neg = 0
all_sents = []

for i, row in data.reset_index().iterrows():
    if type(row.Tags) == str:
        corrected = row.Tokens
        if row.Tokens.isalpha() and row.Tokens.lower() not in known_attributes:
            corrected = spell2.correction(row.Tokens)#spell(row.Tokens)
        if corrected != row.Tokens:
            print(f"{row.Tokens} changed to {corrected}")
        current_sent.append((corrected, row.Tags))
        if "-N-" in row.Tags:
            contains_neg = True
    else:
        all_sents.append(current_sent)
        current_sent = []
        contains_neg = False

all_sents


# In[72]:


for i in range(len(all_sents)):
    print(" ".join(word for word, tag in all_sents[i]))
    print()
    print(" ".join(word + "-" + tag for word, tag in all_sents[i]))


# In[ ]:




