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

import nltk
nltk.download()
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
StopWords = stopwords.words('english')

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import LatentDirichletAllocation


#   Importing data
#   Decision was to focus solely on the text data, then possibly incorporate more features later
path = 'C:/Users/tranw/OneDrive/Documents/Misc/Case studies/Textio Case/Case 2/'

data = pd.read_csv(path + 'kickstarter_corpus.csv')
text = data[['category', 'full_text']]


#   Looking at the distribution of categories
#   Noticed few entries contained null categories, which roughly follow distirbution of categories
cat_n = text['category'].value_counts()
cat_nul = text[text['full_text'].isnull()]['category'].value_counts()
cat_dist = pd.DataFrame({'counts':cat_n, 'nulls':cat_nul}).sort_values('counts', ascending=False)
cat_dist['ratio'] = cat_dist['nulls']/cat_dist['counts']

#   Null entries only about 1% of each category class, so decided to drop them outright
text = text.drop(text[text['full_text'].isnull()].index, axis=0).reset_index(drop=True)


#   Some possible features from text field
text['length'] = text['full_text'].apply(len)
sns.barplot(x='category', y='length', data=text)
len_mean = text.groupby('category').mean()
len_mean[len_mean['length'] > 3000]



#   General method is to perform topic modeling using LDA and use those as features for SVM classifier
#   LDA topics will also allow for more granular interpretation of content in each category
X = text['full_text']
y = text['category']


#   Attempting to tune transforming componenets of pipeline
#   Method is to iteratively split dataset and predict using default classifier
#   Each iteration, a different transforming parameter is used

#   Tuning max words to vectorize from corpus
#   Generally looking for 'elbow' in predictive accuracy, indicating drop off in marginal benefit
#   Computing power is also a factor in consideration when picking final value
scores = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)
    
    x = []
    for j in [1000,2000,4000,8000,12000,16000,20000]:
        
        classifier = Pipeline([
            ('vectorizer', CountVectorizer(max_features=j, analyzer='word', stop_words=StopWords)),
            ('tfidf', TfidfTransformer()),
            ('clf', LinearSVC(multi_class='ovr'))]) 
    
        classifier.fit(X_train, y_train)
        x.append(classifier.score(X_test, y_test))
        print(i,j)
    
    scores.append(x)
    
feat_scores = pd.DataFrame(scores, columns=[1000,2000,4000,8000,12000,16000,20000])
plt.plot(feat_scores.mean())
feat_scores.to_csv(path + 'feat_scores.csv')
n_feat = 8000


#   Tuning number of topics for LDA to draw from
#   Attempted to use perplexity measure to tune, but it seems to be broken in sklearn
scores = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

    x = []
    for j in [10,20,30,50,100,200]:
        
        classifier = Pipeline([
            ('vectorizer', CountVectorizer(max_features=n_feat, analyzer='word', stop_words=StopWords)),
            ('lda', LatentDirichletAllocation(n_topics=j, n_jobs=-1)),
            ('clf', LinearSVC(multi_class='ovr'))])
    
        classifier.fit(X_train, y_train)
        x.append(classifier.score(X_test, y_test))
        print(i,j)
        
    scores.append(x)
    
topic_scores = pd.DataFrame(scores, columns=[10,20,30,50,100,200])
plt.plot(topic_scores.mean())
topic_scores.to_csv(path + 'topic_scores')
n_topic = 50


#   Tuning the final classifier
#   This includes additional parameters for the LDA joint distributions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1000, stratify=y)

classifier = Pipeline([
        ('vectorizer', CountVectorizer(max_features=n_feat, analyzer='word', stop_words=StopWords)),
        ('lda', LatentDirichletAllocation(n_topics=n_topic, n_jobs=-1)),
        ('clf', LinearSVC(multi_class='ovr'))])

params = {#'lda__doc_topic_prior':[None,0.001,0.01,0.1,1],
          #'lda__topic_word_prior':[None,0.001,0.01,0.1,1],
          'clf__C':[0.001,0.01]
          }

gs = GridSearchCV(classifier, param_grid=params, cv=5, n_jobs=-1)
gs.fit(X_train, y_train)
gs.score(X_test, y_test)



cv = gs.best_estimator_.named_steps['vectorizer']
ld = gs.best_estimator_.named_steps['lda']  
sv = gs.best_estimator_.named_steps['clf'] 

topic_dict = {}
for i, j in enumerate(ld.components_):
    topic_dict[i] = ' '.join([cv.get_feature_names()[k] for k in j.argsort()[:-20:-1]])
topics = pd.DataFrame(topic_dict, index=['Words']).T


cat_dict = {}
for i in range(len(sv.classes_)):
    cats[sv.classes_[i]] = sv.coef_[i]
categories = pd.DataFrame(cats)


[x for x in categories['Art'].argsort()].index(1)
categories['Food'][categories['Food'] >= 0.01]
categories['Art'].nsmallest(3)




'''
perplex = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)
    x = []
    for j in range(5,101,5):
        vectorizer = CountVectorizer(max_features=2000, analyzer='word', stop_words=StopWords)
        v = vectorizer.fit_transform(X_train)
        
        lda = LatentDirichletAllocation(n_topics=j, n_jobs=-1)
        lda.fit(v)
        x.append(lda.perplexity(vectorizer.fit_transform(X_test)))
    perplex.append(x)
        
topic_perplex = pd.DataFrame(perplex, columns=range(5,101,5))
plt.plot(topic_perplex.mean())
'''


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print "Topic %d:" % (topic_idx)
        print " ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])


def gen_classes():
len(gs.best_estimator_.named_steps['clf'].coef_)

for i, j in enumerate(sv.classes_):
    print i
    print j

coeffs_mean = pd.DataFrame(gs.best_estimator_.named_steps['clf'].coef_, columns = X.columns).T.mean(axis=1)





