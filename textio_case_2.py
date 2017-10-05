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

path = 'C:/Users/tranw/OneDrive/Documents/Misc/Case studies/Textio Case/Case 2/'

data = pd.read_csv(path + 'kickstarter_corpus.csv')
text = data[['category', 'full_text']]

cat_n = text['category'].value_counts()
cat_nul = text[text['full_text'].isnull()]['category'].value_counts()
cat_dist = pd.DataFrame({'counts':cat_n, 'nulls':cat_nul}).sort_values('counts', ascending=False)
cat_dist['ratio'] = cat_dist['nulls']/cat_dist['counts']

text = text.drop(text[text['full_text'].isnull()].index, axis=0).reset_index(drop=True)

text['length'] = text['full_text'].apply(len)
sns.barplot(x='category', y='length', data=text)
len_mean = text.groupby('category').mean()
len_mean[len_mean['length'] > 3000]


X = text['full_text']
y = text['category']



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
n_feat = 8000
#feat_rng = range(5000, 10001, 500)


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
test_topic = 


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


classifier = Pipeline([
        ('vectorizer', CountVectorizer(max_features=test_feat, analyzer='word', stop_words=StopWords)),
        ('lda', LatentDirichletAllocation(n_topics=j, n_jobs=-1)),
        ('clf', LinearSVC(multi_class='ovr'))])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1000, stratify=y)
classifier = Pipeline([
        ('vectorizer', CountVectorizer(max_features=n_feat, analyzer='word', stop_words=StopWords)),
        ('lda', LatentDirichletAllocation(n_jobs=-1)),
        ('clf', LinearSVC(multi_class='crammer-singer'))])
params = {'lda__n_topics':[10,20,30,50,100,200],
          'lda__doc_topic_prior':[None,0.001,0.01,0.1,1],
          'lda__topic_word_prior':[None,0.001,0.01,0.1,1],
          'clf__penalty':['l1','l2'],
          'clf__C':[0.001,0.01,1,10,100]}

gs = GridSearchCV(classifier, param_grid=params, cv=5, n_jobs=-1)
gs.fit(X_train, y_train)
gs.score(X_test, y_test)




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

vectorizer = CountVectorizer(max_features=8000, analyzer='word', stop_words=StopWords)
v = vectorizer.fit_transform(X_train)

lda = LatentDirichletAllocation(n_topics=20, n_jobs=-1)
lda.fit(v)

lda.components_
feat_names = vectorizer.get_feature_names()

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print "Topic %d:" % (topic_idx)
        print " ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])

display_topics(lda, feat_names, 20)













test = text['full_text'][0]

tk_words = RegexpTokenizer('\w+')
tk_nonwords = RegexpTokenizer('\W+')


z = tk_words.tokenize(test)
tk_nonwords.tokenize(test)


classifier = Pipeline([
    ('vectorizer', CountVectorizer(max_features=1000, analyzer='word', stop_words=StopWords)),
#    ('tfidf', TfidfTransformer()),
    ('lda', LatentDirichletAllocation(n_topics=30, n_jobs=-1)),
    ('clf', LinearSVC(multi_class='crammer_singer'))])

classifier.fit(text['full_text'].iloc[0:5000], text['category'].iloc[0:5000])
pred = classifier.predict(text['full_text'].iloc[5000:])


y_test = text['category'][5000:]
y_test.reset_index(inplace=True, drop=True)

predictions = pd.DataFrame([pred, y_test, text['full_text'][5000:]]).T
#predictions.to_csv(path + 'test.csv')

yes = 0
for i in range(len(predictions)):
    if predictions[0].iloc[i] == predictions[1].iloc[i]:
        yes += 1
print(float(yes)/len(y_test))\


mlb.inverse_transform(pred)
sum(pred == y[5000:])/len(y[5000:])

cntvec = CountVectorizer(max_features=1000, analyzer='word', stop_words=StopWords)
x = cntvec.fit([text['full_text'][0]])
z = TfidfTransformer()
f = z.fit_transform(x)



text.info()
len(data['category'].value_counts())
data[data['full_text'].isnull()]['blurb']

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(text['category'])





