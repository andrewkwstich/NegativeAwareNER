#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import json
from pandas import json_normalize
import gzip


# In[2]:


import pandas as pd

attributes = {}
def getDF(path):
    i = 0
    df = {}
    for d in open(path):
        json_str = json.loads(d)
        if "style" not in json_str:
            continue
        for k,v in json_str["style"].items():
            if k not in attributes:
                attributes[k] = set()
            attributes[k].add(v.strip()) 
        if 'reviewText' not in json_str:
            continue
        df[i] = json_str["reviewText"]
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

df_home = getDF('../data/Home_and_Kitchen_5.json')
df_tools = getDF('../data/Tools_and_Home_Improvement_5.json')


# In[6]:


df_home.to_csv('../data/reviews_home.csv')


# In[5]:


df_tools.to_csv('../data/reviews_tools.csv')


# In[32]:


attributes.keys()


# In[3]:


attributes["Color:"]


# In[4]:


attributes["Colour:"]


# In[18]:


attributes["Format:"]


# In[33]:


import csv
with open('../data/attributes.csv', 'w', newline='\n') as csvfile:
    for attr, val_set in attributes.items():
        for val in val_set:
            csvfile.write(val)
            csvfile.write('\n')


# In[ ]:




