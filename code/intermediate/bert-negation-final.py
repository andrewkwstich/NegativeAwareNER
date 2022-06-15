#!/usr/bin/env python
# coding: utf-8

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## Changing the transformers and tokenizer versions

# In[2]:


# pip install tokenizers==0.8.0rc4

# pip install transformers==3.0.1


import transformers
import tokenizers


# ### Imports
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertConfig, BertForTokenClassification
from transformers import AutoTokenizer, AutoModelForTokenClassification,AutoConfig


from torch import cuda
from sklearn.metrics import f1_score, classification_report
import csv
# ### Reading the SOCC data

class dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
            self.len = len(dataframe)
            self.data = dataframe
            self.tokenizer = tokenizer
            self.max_len = max_len

    def __getitem__(self, index):
            # step 1: get the sentence and word labels 
            sentence = self.data.Sentence[index].strip().split()  
            word_labels = self.data.Negation[index].split(",") 

            # step 2: use tokenizer to encode sentence (includes padding/truncation up to max length)
            # BertTokenizerFast provides a handy "return_offsets_mapping" functionality for individual tokens
            encoding = self.tokenizer(sentence,
                                # is_pretokenized=False, 
                                return_offsets_mapping=True, 
                                padding='max_length', 
                                truncation=True, 
                                max_length=self.max_len)
            
            labels_to_ids = {'True':1,'False':0}
            ids_to_labels = {1: 'True', 0: 'False'}
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

def main():

    
    negation_data_socc= pd.read_csv("data/SOCC_negation.tsv", sep = "\t")


    print(negation_data_socc.info())


    negation_data_socc['Negation'] = negation_data_socc['Negation'].map({True: 'True', False: 'False'})

    negation_data_socc['Sentence']= negation_data_socc[['Sentence_index','Token','Negation']].groupby(['Sentence_index'])['Token'].transform(lambda x: ' '.join(x))

    negation_data_socc['Negation']= negation_data_socc[['Sentence_index','Token','Negation']].groupby(['Sentence_index'])['Negation'].transform(lambda x: ','.join(x))


    negation_data_socc = negation_data_socc[['Sentence','Negation']]


    negation_data_socc = negation_data_socc.drop_duplicates().reset_index(drop=True)



    # ### Tokenization


    labels_to_ids = {'True':1,'False':0}
    ids_to_labels = {1: 'True', 0: 'False'}


    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 4
    VALID_BATCH_SIZE = 2
    EPOCHS = 1
    LEARNING_RATE = 1e-05
    MAX_GRAD_NORM = 10
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    # print(tokenizers.__version__)

    train_size = 0.8
    train_dataset = negation_data_socc.sample(frac=train_size,random_state=200)
    test_dataset = negation_data_socc.drop(train_dataset.index).reset_index(drop=True)
    test_dataset.drop(test_dataset.index[[233, 234, 457]], inplace=True)
    test_dataset = test_dataset.reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)
    train_dataset.drop(train_dataset.index[[573, 1506, 1603, 1700]], inplace=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(negation_data_socc.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    training_set = dataset(train_dataset, tokenizer, MAX_LEN)
    testing_set = dataset(test_dataset, tokenizer, MAX_LEN)



    external_valset= pd.read_csv("data/un-negated_clean_data.csv")

    external_valset["Sentence"]= external_valset[["Tokens","is_negative","sentence"]].groupby(["sentence"])["Tokens"].transform(lambda x: ' '.join(x))


    external_valset['is_negative'] = external_valset['is_negative'].map({True: 'True', False: 'False'})

    external_valset["Negation"]= external_valset[["Tokens","is_negative","sentence"]].groupby(["sentence"])["is_negative"].transform(lambda x: ','.join(x))

    external_valset = external_valset[['Sentence','Negation']]

    external_valset = external_valset.drop_duplicates().reset_index(drop=True)


    final_validation_set = dataset(external_valset, tokenizer, MAX_LEN)


    # ### Loading the BERT Model



    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': False,
                    'num_workers': 0
                    }

    # print(training_set.__len__())
    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)



    valid_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': False,
                    'num_workers': 0
                    }
    valid_loader = DataLoader(final_validation_set, **valid_params)


    device = 'cuda' if cuda.is_available() else 'cpu'
    print(device)


    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(labels_to_ids))
    model.to(device)


    inputs = training_set[2]
    input_ids = inputs["input_ids"].unsqueeze(0)
    attention_mask = inputs["attention_mask"].unsqueeze(0)
    labels = inputs["labels"].unsqueeze(0)

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    outputs, _ = model(input_ids, attention_mask=attention_mask, labels=labels)
    initial_loss = outputs[0]
    initial_loss


    tr_logits = outputs[1]
    tr_logits.shape

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


    # ### Training and fine-tuneing the BERT model using the SOCC dataset


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



    for epoch in range(EPOCHS):
        print(f"Training epoch: {epoch + 1}")
        train(epoch)


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


    # ### Validating the BERT model on the test split of the SOCC dataset


    labels, predictions = valid(model, testing_loader)


    print(classification_report(labels, predictions, labels = list(labels_to_ids.keys()) ))


    # ### Validating the BERT model on the HeyDay Validation set.


    labels, predictions = valid(model, valid_loader)

    print(classification_report(labels, predictions, labels = list(labels_to_ids.keys()) ))


    # ### Writing into the CSV file and generating the final tags


    valset= pd.read_csv("data/Heyday_validationdata.csv")


    token_list = valset.Tokens.values.tolist()
    sentence_list = valset.Sentence_Index.values.tolist()
    NER_list = valset.Predicted_NER_Tags.values.tolist()




    rows = zip(sentence_list,token_list,NER_list,predictions)
    with open('data/Heyday_predvalidationdata.csv', "w") as f:
        writer = csv.writer(f)
        writer.writerow(('Sentence_Index','Tokens','Predicted_NER_Tags','Predicted_Negation'))
        for row in rows:
            writer.writerow(row)


    final_valset= pd.read_csv("data/Heyday_final_predvalidationdata.csv")


    final_valset = final_valset.replace('I-','', regex=True)


    def label_final (row):
        if row['Predicted_NER_Tags'] == 'O' :
            return 'O'
        if row['Predicted_NER_Tags'] != 'O' and row['Predicted_Negation'] == True :
            return "I-N-"+row['Predicted_NER_Tags']
        if row['Predicted_NER_Tags'] != 'O' and row['Predicted_Negation'] == False :
            return "I-"+row['Predicted_NER_Tags']


    final_valset['final_label'] = final_valset.apply (lambda row: label_final(row), axis=1)



    # ### Validating the final results



    clean_data = pd.read_csv("data/clean_data.csv")

    clean_data = clean_data.replace('B-','I-', regex=True)

    clean_list = clean_data.Tags.values.tolist()
    pred_list = final_valset.final_label.values.tolist()


    print(classification_report(clean_list, pred_list))


    # for i in zip(tokens,predictions):
    #     print(i)

if __name__ == "__main__":
    main()