#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## Changing the transformers and tokenizer versions

# In[2]:


pip install tokenizers==0.8.0rc4


# In[3]:


pip install transformers==3.0.1


# In[4]:


import transformers
transformers.__version__


# In[5]:


import tokenizers
tokenizers.__version__


# ### Imports

# In[7]:


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertConfig, BertForTokenClassification
from transformers import AutoTokenizer, AutoModelForTokenClassification,AutoConfig


# ## Reading data

# In[8]:


un_neg_data= pd.read_csv("../input/nertag/unnegated_File_Name.csv")


# In[9]:


un_neg_data


# In[7]:



data = pd.read_csv("../input/nertag/File_Name.csv")


# In[8]:


data


# In[69]:


secondary_data = pd.read_csv("../input/nertag/clean_data.csv")


# In[10]:


unneg_secondary_data= pd.read_csv("../input/nertag/un-negated_clean_data.csv")


# In[11]:


unneg_secondary_data


# In[12]:


unneg_secondary_data['Sentence']= unneg_secondary_data[['sentence','Tokens','is_negative']].groupby(['sentence'])['Tokens'].transform(lambda x: ' '.join(x))


# In[13]:


unneg_secondary_data['is_negative'] = unneg_secondary_data['is_negative'].map({True: 'True', False: 'False'})


# In[14]:


unneg_secondary_data['is_negative']= unneg_secondary_data[['sentence','Tokens','is_negative']].groupby(['sentence'])['is_negative'].transform(lambda x: ','.join(x))


# In[15]:


unneg_secondary_data


# In[16]:


unneg_secondary_data = unneg_secondary_data[['Sentence','is_negative']]


# In[17]:


unneg_secondary_data = unneg_secondary_data.drop_duplicates().reset_index(drop=True)


# In[18]:


unneg_secondary_data


# In[19]:


unneg_secondary_data.to_csv(r'./unnegated_Fil.csv', index = False)


# In[10]:


secondary_data


# ### Tokenization

# In[24]:


labels_to_ids = {k: v for v, k in enumerate(unneg_secondary_data.Tags.unique())}
ids_to_labels = {v: k for v, k in enumerate(unneg_secondary_data.Tags.unique())}
labels_to_ids


# In[20]:


labels_to_ids = {'True':1,'False':0}
ids_to_labels = {1: 'True', 0: 'False'}


# In[21]:


MAX_LEN = 128
TRAIN_BATCH_SIZE = 128
VALID_BATCH_SIZE = 2
EPOCHS = 1
LEARNING_RATE = 1e-05
MAX_GRAD_NORM = 10
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


# In[25]:


class dataset(Dataset):
  def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

  def __getitem__(self, index):
        # step 1: get the sentence and word labels 
        sentence = self.data.Sentence[index].strip().split()  
        word_labels = self.data.is_negative[index].split(",") 

        # step 2: use tokenizer to encode sentence (includes padding/truncation up to max length)
        # BertTokenizerFast provides a handy "return_offsets_mapping" functionality for individual tokens
        encoding = self.tokenizer(sentence,
                             is_pretokenized=True, 
                             return_offsets_mapping=True, 
                             padding='max_length', 
                             truncation=True, 
                             max_length=self.max_len)
        
        # step 3: create token labels only for first word pieces of each tokenized word
        labels = [labels_to_ids[label] for label in word_labels] 
        # code based on https://huggingface.co/transformers/custom_datasets.html#tok-ner
        # create an empty array of -100 of length max_length
        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        
        # set only labels whose first offset position is 0 and the second is not 0
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
          if mapping[0] == 0 and mapping[1] != 0:
            # overwrite label
            encoded_labels[idx] = labels[i]
            i += 1

        # step 4: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(encoded_labels)
        
        return item

  def __len__(self):
        return self.len


# In[ ]:


# tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER",use_fast=True)
# model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")


# In[26]:


train_size = 0.8
train_dataset = unneg_secondary_data.sample(frac=train_size,random_state=200)
test_dataset = unneg_secondary_data.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

print("FULL Dataset: {}".format(unneg_secondary_data.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = dataset(train_dataset, tokenizer, MAX_LEN)
testing_set = dataset(test_dataset, tokenizer, MAX_LEN)


# In[ ]:





# In[27]:


training_set[0]


# In[28]:


for token, label in zip(tokenizer.convert_ids_to_tokens(training_set[0]["input_ids"]), training_set[0]["labels"]):
  print('{0:10}  {1}'.format(token, label))


# In[29]:


train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)


# In[30]:


from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
print(device)


# ### Model Generation

# In[37]:


model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(labels_to_ids))
# model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
model.to(device)


# In[ ]:


# from transformers import BertForSequenceClassification, BertConfig

# config = AutoConfig.from_pretrained("dslim/bert-base-NER")
# config.num_labels = 28
# model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER", config =config)
# model.parameters


# In[38]:


inputs = training_set[2]
input_ids = inputs["input_ids"].unsqueeze(0)
attention_mask = inputs["attention_mask"].unsqueeze(0)
labels = inputs["labels"].unsqueeze(0)

input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)
labels = labels.to(device)

outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
initial_loss = outputs[0]
initial_loss


# In[39]:


tr_logits = outputs[1]
tr_logits.shape


# In[40]:


optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


# In[41]:



def train(epoch):
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    # put model in training mode
    model.train()
    
    for idx, batch in enumerate(training_loader):
        
        ids = batch['input_ids'].to(device, dtype = torch.long)
        mask = batch['attention_mask'].to(device, dtype = torch.long)
        labels = batch['labels'].to(device, dtype = torch.long)

        loss, tr_logits = model(input_ids=ids, attention_mask=mask, labels=labels)
        tr_loss += loss.item()

        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)
        
        if idx % 100==0:
            loss_step = tr_loss/nb_tr_steps
            print(f"Training loss per 100 training steps: {loss_step}")
           
        # compute training accuracy
        flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
        active_logits = tr_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
        
        # only compute accuracy at active labels
        active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        #active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))
        
        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)
        
        tr_labels.extend(labels)
        tr_preds.extend(predictions)

        tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy
    
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=MAX_GRAD_NORM
        )
        
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    print(f"Training loss epoch: {epoch_loss}")
    print(f"Training accuracy epoch: {tr_accuracy}")


# In[42]:


for epoch in range(EPOCHS):
    print(f"Training epoch: {epoch + 1}")
    train(epoch)


# In[43]:


def valid(model, testing_loader):
    # put model in evaluation mode
    model.eval()
    
    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []
    
    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):
            
            ids = batch['input_ids'].to(device, dtype = torch.long)
            mask = batch['attention_mask'].to(device, dtype = torch.long)
            labels = batch['labels'].to(device, dtype = torch.long)
            
            loss, eval_logits = model(input_ids=ids, attention_mask=mask, labels=labels)
            
            eval_loss += loss.item()

            nb_eval_steps += 1
            nb_eval_examples += labels.size(0)
        
            if idx % 100==0:
                loss_step = eval_loss/nb_eval_steps
                print(f"Validation loss per 100 evaluation steps: {loss_step}")
              
            # compute evaluation accuracy
            flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
            active_logits = eval_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
            
            # only compute accuracy at active labels
            active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        
            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)
            
            eval_labels.extend(labels)
            eval_preds.extend(predictions)
            
            tmp_eval_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            eval_accuracy += tmp_eval_accuracy

    labels = [ids_to_labels[id.item()] for id in eval_labels]
    predictions = [ids_to_labels[id.item()] for id in eval_preds]
    
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    print(f"Validation Loss: {eval_loss}")
    print(f"Validation Accuracy: {eval_accuracy}")

    return labels, predictions


# In[44]:


labels, predictions = valid(model, testing_loader)


# In[46]:


# from seqeval.metrics import classification_report
from sklearn.metrics import f1_score, classification_report

print(classification_report(labels, predictions, labels = list(labels_to_ids.keys()) ))

