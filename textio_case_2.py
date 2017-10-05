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
from sklearn.svm import LinearSVC
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
    for j in range(1000,10001,500):
        classifier = Pipeline([
            ('vectorizer', CountVectorizer(max_features=10000, analyzer='word', stop_words=StopWords)),
            ('tfidf', TfidfTransformer()),
            ('clf', LinearSVC(multi_class='crammer_singer'))])
        classifier.fit(X_train, y_train)
        x.append(classifier.score(X_test, y_test))
        #print(i,j)
    scores.append(x)

feat_scores = pd.DataFrame(scores, columns=range(1000,10001,500))
#plt.plot(feat_scores.mean())


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
        
topic_perplex = pd.DataFrame(scores, columns=range(1000,10001,500))
#plt.plot(topic_perplex.mean())




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





