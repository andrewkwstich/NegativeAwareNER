#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
data = pd.read_csv('data/ner_tagged.tsv', sep='\t')


# In[9]:


data


# In[10]:


data.rename(columns = {'O':'Tags','-DOCSTART-':'Tokens','-X-':'X'}, inplace = True)


# In[11]:


data


# In[12]:


for i, row in data.reset_index().iterrows():
    if type(row.Tokens) == str:
        if row.Tokens.startswith("un") and "-N-" in row.Tags and row.Tokens != "undermounted":
            data.at[i,'Tags'] = row.Tags.replace("N-", "")
    if type(row.Tokens) == str:
        if row.Tokens.endswith("less") or row.Tokens.endswith("less.") and row.Tokens not in ["screw", "less"] and "-N-" in row.Tags:
            data.at[i,'Tags'] = row.Tags.replace("N-", "")


# In[13]:


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


# In[14]:


data["is_negative"] = is_negative


# In[15]:


get_ipython().system('pip install pandas==1.3.0')


# In[16]:


import pandas
data.to_csv("data/reannotated_val.csv")


# In[17]:


neg_sents = []
pos_sents = []
current_sent = []
current_tags = []
contains_neg = False
non_neg = 0

for i, row in data.reset_index().iterrows():
    if type(row.Tags) == str:
        current_sent.append((row.Tokens, row.Tags, row.is_negative))
        if row.is_negative:
            contains_neg = True
    else:
        if contains_neg:
            neg_sents.append(current_sent)
        else:
            pos_sents.append(current_sent)
        current_sent = []
        contains_neg = False


# In[18]:


for entry in neg_sents:
    print(entry)


# In[19]:


for entry in pos_sents:
    for word in entry:
        if word[0].startswith("un") and "-N-" in word[1]:
            print(entry)
        if word[0].endswith("less"):
            print(entry)


# ## Categories
# 
# - bathroom (sink, mirror, shower, bathtub, shower curtains, squeegee, brush, toilet)
# - blankets
# - mirror
# - clothing and accessories
# - vanity/shelving/wardrobes
# - kitchen (sink, dishes, island)
# - glass doors
# - gazebo
# - supplements
# - luggage
# - furniture
# - pool cue case
# - barbecue

# ## Observations
# 
# - Difficult to locate and distinguish products and attributes even prior to attempting negation. Would it make sense to figure out what attributes likely to belong to a particular class of products are?
# - Negation tends not to persist across sentence boundaries
# - Punctuation is not separately tokenized
# - Annotation of negative size is not perfectly consistent: in the 17th negative sentence, the height is negated ('15" height'), whereas in the 14th sentence, the comparative is negated ("bigger").
# - In words where the negation is marked by a prefix, e.g. "unflavored" in "I'm looking for some unflavored mass gainer protein", the entire word is given a negative tag ("unflavored-B-N-ATTRIBUTE"). This seems like it could cause confusion, since in this case it implies that "unflavored" is not what is desired.
# - The "direction" of negation is not always clear from the annotation. E.g., in "I found a vanity top I love but it has only one hole for taps and that doesn't suit our needs", "one hole" is negated, but it is not clear from that that more rather than fewer holes are desired.
# - Lots of grammatical errors, e.g. "with not soy in it". Using word and character embeddings would help with this
# - "doesn't need to be X-attribute" is annotated with a negation
# - 
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
# - "the website only gives the option of"
# - "without X or Y" (two separately annotated entities separated by conjunction)

# In[ ]:




