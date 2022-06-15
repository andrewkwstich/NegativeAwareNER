#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[16]:


tax_file = "../data/taxonomy.txt"


# In[17]:


f = open(tax_file)


# In[18]:


import csv
with open('../data/products.csv', 'w', newline='\n') as csvfile:
    for line in f:
        print(line)
        val = line.split('>')[-1].strip()
        csvfile.write(val)
        csvfile.write('\n')


# In[ ]:




