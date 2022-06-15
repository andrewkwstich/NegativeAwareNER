#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data = pd.read_csv('./ner_tagged.tsv', sep='\t')


# In[4]:


data.rename(columns = {'Unnamed: 3':'Tags','-DOCSTART-':'Tokens','-X-':'X'}, inplace = True)


# In[82]:


neg_sents = []
current_sent = []
current_tags = []
contains_neg = False
non_neg = 0

for i, row in data.reset_index().iterrows():
    if type(row.Tags) == str:
        current_sent.append((row.Tokens, row.Tags))
        if "-N-" in row.Tags:
            contains_neg = True
    else:
        if contains_neg:
            neg_sents.append(current_sent)
        else:
            non_neg += 1
        current_sent = []
        contains_neg = False

neg_sents


# In[83]:


non_neg


# In[119]:


i = 46


# In[120]:


i += 1

print(" ".join(word for word, tag in neg_sents[i]))
print()
print(" ".join(word + "-" + tag for word, tag in neg_sents[i]))

print(i)


# ## Observations
# 
# - Difficult to locate and distinguish products and attributes even prior to attempting negation. Would it make sense to figure out what attributes likely to belong to a particular class of products are?
# - Negation tends not to persist across sentence boundaries
# - Punctuation is not separately tokenized
# - Annotation of negative size is not perfectly consistent: in the 17th negative sentence, the height is negated ('15" height'), whereas in the 14th sentence, the comparative is negated ("bigger").
# - In words where the negation is marked by a prefix, e.g. "unflavored" in "I'm looking for some unflavored mass gainer protein", the entire word is given a negative tag ("unflavored-B-N-ATTRIBUTE"). This seems like it could cause confusion, since in this case it implies that "unflavored" is not what is desired.
# - The "direction" of negation is not always clear from the annotation. E.g., in "I found a vanity top I love but it has only one hole for taps and that doesn't suit our needs", "one hole" is negated, but it is not clear from that that more rather than fewer holes are desired.
# - Lots of grammatical errors, e.g. "with not soy in it". Using word embeddings would help with this
# 
# ## Ways of indicating negation (incomplete):
# 
# - "don't want (any)"
# - "no"
# - "no more than"
# - "less than"
# - "un-"
# - "without"
# - "only"
# - "-less"
# - "instead of"
# - "too" (what comes before is negated, e.g. in "pink is too light", "pink" is negated)
# - "not too" (what comes after is negated, e.g. in "not too light", "light" is negated)
