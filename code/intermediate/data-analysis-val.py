#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    data = pd.read_csv('data/ner_tagged.tsv', sep='\t')

    data.rename(columns = {'Unnamed: 3':'Tags','-DOCSTART-':'Tokens','-X-':'X'}, inplace = True)
    data.X.unique()
    data.O.unique()
    data.info()

    data.describe(include = 'all')

    data = data.drop(columns= ['X','O'])
    data.head()


    
    print(data['Tags'].isna().sum())

    data["Tags"].value_counts(normalize=True)

    data["Tags"].value_counts()


    plt.figure(figsize=(15, 5))
    ax = sns.countplot('Tags', data=data.loc[data['Tags'] != 'O'])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")
    plt.tight_layout()
    plt.show()

    print(data.loc[data['Tags'] == 'B-PRODUCT', 'Tokens'].head())

if __name__ == "__main__":
    main()