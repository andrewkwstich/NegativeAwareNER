#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install --user flair')


# In[1]:


from flair.data import Sentence
from flair.models import SequenceTagger

# load tagger
tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")



# In[ ]:




