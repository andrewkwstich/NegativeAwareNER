
import pandas as pd
import os
import nltk
from spacy.tokenizer import Tokenizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import f1_score, classification_report
from bs4 import BeautifulSoup
from nltk.corpus import names,gazetteers 
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_f1_score, flat_classification_report
import re

import sklearn_crfsuite
import csv
from spacy.lang.en import English
import spacy
import contextualSpellCheck

crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=350,
        all_possible_transitions=True,
        verbose = True
    )


# ### Creating features using knowledge bases

colors = pd.read_csv("data/colors.csv")


color_list= []
for index, row in colors.iterrows():
    color_list.append(row['Air Force Blue (Raf)'])


products = pd.read_csv('data/products.csv',sep='\t')

product_list =[]
for index, row  in products.iterrows():
    for i in row[0].split():
        if i[-1]=='s':
            product_list.append(i[:-1])
        else:
            product_list.append(i)


product_list = set(product_list)
product_list = list(product_list)


# ### Feature Engineering


def get_pos(word):
    tag = nltk.pos_tag([word])
    return tag[0][1]

def is_number(string):
    return any(char.isdigit() for char in string)

def word2features(sentence, idx):
    word_features = {}
    word_features['word_lowercase'] = sentence[idx].lower()

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
    
    
    ## Extra Features:
    
    ## Is Number
    word_features["number"] = is_number(sentence[idx])
    
    ##Is_Noun
    
    if get_pos(sentence[idx])== "NN":
        word_features["is_noun"] = True
    else:
        word_features["is_noun"] = False
    

#     color Feature:
    
    if sentence[idx].lower() in color_list:
        word_features["color"] = True
    else:
        word_features["color"] = False
        
#     Product Feature:
    
    if sentence[idx].lower() in product_list:
        word_features["product"] = True
    else:
        word_features["product"] = False



        
    for i in range(len(sentence)):
        if i == idx and i != 0:
            word_features['first_word_not_in_title_case'] = sentence[idx].istitle()
        elif i == idx and i == 0:
            if sentence[idx].istitle():
                word_features['first_word_not_in_title_case'] = False
    
        
    return word_features
    
    
def sentence2features(sentence):
    return [word2features(sentence, idx) for idx in range(len(sentence))]

def main():
    
    data= pd.read_csv('data/Heyday_trainingdata.csv')

    data['Sentence']= data[['Sentence_Index','Tokens','Tags']].groupby(['Sentence_Index'])['Tokens'].transform(lambda x: ' '.join(x))


    data['Tags']= data[['Sentence_Index','Tokens','Tags']].groupby(['Sentence_Index'])['Tags'].transform(lambda x: ','.join(x))


    data = data[['Sentence','Tags']]

    data = data.drop_duplicates().reset_index(drop=True)

    # ### Loading the training and validation set

    train_dataset= data
    test_dataset = pd.read_csv('data/un-negated_clean_data.csv')
    test_dataset = test_dataset[["Tokens","Tags","sentence"]]
    test_dataset = test_dataset.replace('B-','I-', regex=True)
    test_dataset['Sentence']= test_dataset[['sentence','Tokens','Tags']].groupby(['sentence'])['Tokens'].transform(lambda x: ' '.join(x))
    test_dataset['Tags']= test_dataset[['sentence','Tokens','Tags']].groupby(['sentence'])['Tags'].transform(lambda x: ','.join(x))
    test_dataset = test_dataset[['Sentence','Tags']]
    test_dataset = test_dataset.drop_duplicates().reset_index(drop=True)


    def strsplit_tags(tags):
        sent_list = tags.split(",")
        return sent_list



    def strsplit_sentence(sentence):
        sent_list = sentence.split(" ")
        return sent_list

    train_sents = []

    for index, row in train_dataset.iterrows():
        train_sents.append((strsplit_sentence(row['Sentence']),strsplit_tags(row['Tags'])))


    dev_sents = []

    for index, row in test_dataset.iterrows():
        dev_sents.append((strsplit_sentence(row['Sentence']),strsplit_tags(row['Tags'])))

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


    train_dicts, train_tags = prepare_ner_feature_dicts(train_sents)
    dev_dicts, dev_tags = prepare_ner_feature_dicts(dev_sents)

    # ### CRF Model

    
    crf.fit(train_dicts, train_tags)
    try:
        call_produces_an_error()
    except:
        pass


    # ### Results on the validation set


    def flatten(l):
        result = []
        for sub in l:
            result.extend(sub)
        return result

    y_pred = crf.predict(dev_dicts)
    print(f1_score(flatten(dev_tags), flatten(y_pred), average='macro'))
    print(f1_score(flatten(dev_tags), flatten(y_pred), average='micro'))
    print(classification_report(flatten(dev_tags), flatten(y_pred)))


    #Convert a sentence into feature dictionary

    # print(dev_sents)
   
    # ### Tagging the Validation set

    val_set= pd.read_csv('data/un-negated_clean_data.csv')

    token_list = val_set.Tokens.values.tolist()
    sentence_list = val_set.sentence.values.tolist()


    rows = zip(sentence_list,token_list,flatten(y_pred))
    with open('data/Heyday_validationdata.csv', "w") as f:
        writer = csv.writer(f)
        writer.writerow(('Sentence_Index','Tokens','Predicted_NER_Tags'))
        for row in rows:
            writer.writerow(row)

    print(test_dataset)

def predict(sentence, correct_spellings=True):
    nlp = spacy.load('en_core_web_sm')
    contextualSpellCheck.add_to_pipe(nlp)
    input_str = sentence
    doc = nlp(input_str)
    if correct_spellings:
        tokens = [token.text for token in nlp(doc._.outcome_spellCheck)]
    else:
        tokens = [token.text for token in doc]
    
    print(tokens)

    feat = sentence2features(tokens)
    output_str = crf.predict([feat])
    print(output_str)
    result = []
    for token, tag in zip(tokens, output_str[0]):
        result.append((token, tag))
    return result


if __name__ == "__main__":
    main()