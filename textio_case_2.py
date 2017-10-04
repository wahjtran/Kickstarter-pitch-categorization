# -*- coding: utf-8 -*-
"""
Created on Wed Oct 04 11:25:34 2017

@author: tranw
"""



import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns


path = 'C:/Users/tranw/OneDrive/Documents/Misc/Case studies/Textio Case/Case 2/'

data = pd.read_csv(path + 'kickstarter_corpus.csv')
text = data[['category', 'full_text']]

cat_n = text['category'].value_counts()
cat_nul = text[text['full_text'].isnull()]['category'].value_counts()
cat_dist = pd.DataFrame({'counts':cat_n, 'nulls':cat_nul}).sort_values('counts', ascending=False)
cat_dist['ratio'] = cat_dist['nulls']/cat_dist['counts']

text = text.drop(text[text['full_text'].isnull()].index, axis=0).reset_index(drop=True)

text['length'] = text['full_text']




text.info()
len(data['category'].value_counts())
data[data['full_text'].isnull()]['blurb']










