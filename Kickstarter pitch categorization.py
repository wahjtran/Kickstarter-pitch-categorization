# -*- coding: utf-8 -*-
"""
Created on Wed Oct 04 11:25:34 2017

@author: tranw
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
nltk.download()
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
StopWords = stopwords.words('english')

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import confusion_matrix, classification_report, recall_score, roc_auc_score, make_scorer


#   Importing data

path = 'C:/Users/tranw/OneDrive/Documents/Misc/Case studies/Textio Case/Case 2/'
data = pd.read_csv(path + 'kickstarter_corpus.csv')


####	EXPLORATORY ANALYSIS	####

#   Looking at the distribution of categories
#   Noticed few entries contained null categories, which roughly follow distirbution of categories

cat_n = data['category'].value_counts()
cat_nul = data[data['full_text'].isnull()]['category'].value_counts()

cat_dist = pd.DataFrame({'counts':cat_n, 'nulls':cat_nul}).sort_values('counts', ascending=False)
cat_dist['ratio'] = cat_dist['nulls']/cat_dist['counts']

#   Null entries only about 1% of each category class, so decided to drop them outright
data = data.drop(data[data['full_text'].isnull()].index, axis=0).reset_index(drop=True)



#	Looking at some of the other columns
#	Decided to include 'goal' based on some separation in categories
#	Separation is huge for Technology; large std and probably outliers

sns.countplot(x='category', hue='funded', data=data)

sns.barplot(x='category', y='goal', data=data)
goal_mean = data.groupby('category').mean()['goal']



#   Exploring some possible features from text field based on length and specific markers

word = RegexpTokenizer('\w+|\$[\d\.]+')
nonword = RegexpTokenizer('\W+')


#   Length of text field
#	Surprising separation in text length between two groups of category classes

data['len_text'] = data['full_text'].apply(len)
sns.barplot(x='category', y='len_text', data=data)
len_mean = data.groupby('category').mean()['len_text']

len_mean[len_mean > 3000]


#   Number of numeric values in text field
#	High frequency of numeric values may point to quantifiable traits

def count_num(t):
    x = word.tokenize(t)
    y = 0
    for i in x:
        try:
            float(i)
            y += 1
        except ValueError:
            pass
    return y

data['n_num'] = data['full_text'].apply(count_num)
sns.barplot(x='category', y='n_num', data=data)
n_num_mean = data.groupby('category').mean()['n_num']


#   Number of mentions of money (USD)
#	Specifying costs, pricing, or funding levels may be more prevalent in certain categories

def count_money(t):
    x = word.tokenize(t)
    y = 0
    for i in x:
        y += (i[0] == '$')
    return y

data['n_money'] = data['full_text'].apply(count_money)
sns.barplot(x='category', y='n_money', data=data)
n_money_mean = data.groupby('category').mean()['n_money']


#   Content density of text: how many non-stopwords compared to total words in text
#	No separation observed, but noticed non-null text fields with no actual words

def stop_ratio(t):
    x = word.tokenize(t)
    try:
        return len([w for w in x if w.lower() not in StopWords])/float(len(x))
    except ZeroDivisionError:
        return float(0)

data['content_ratio'] = data['full_text'].apply(stop_ratio)
sns.barplot(x='category', y='content_ratio', data=data)

#	Drop additional rows with no words
data = data.drop(data[data['content_ratio'] == 0].index, axis=0).reset_index(drop=True)


#   Length of blurb field
#	No separation observed

data['len_blurb'] = data['blurb'].apply(len)
sns.barplot(x='category', y='len_blurb', data=data)


#	Output potentially relevant fields for graphing

pd.DataFrame([goal_mean, len_mean, n_num_mean, n_money_mean]).T.to_csv(path + 'other_features.csv')



####	DATA PREPARATION	####

#	General methodology is to use LDA to create topic features to use in SVM modeling algorithm
#	Count vectorizer and LDA must be tuned to optimally generate topics and can only process text vector
#	After tuning count vectorizer and LDA and transforming text field to topic vectors, non-text columns will be reintroduced
#	SVM classifer will then be tuned on the full dataset


#	Drop unused columns to assemble modeling dataset

model_data = data.drop(['author', 'backers', 'blurb', 'end_date', 'location',
                        'pledged', 'title', 'len_blurb', 'content_ratio'], axis=1)

X = model_data.drop(['category', 'funded'], axis=1)
y = model_data[['category', 'funded']]


#	Split data into training and testing sets
#	Text field must be further separated to accomodate count vectorizer and LDA algorithm

seed = 888
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1000, stratify=y, random_state=seed)

X_train_text = X_train['full_text']
X_test_text = X_test['full_text']


#	Primary target is category, but it might be useful to check relationship of topics with funding success

y_train_category = y_train['category']
y_test_category = y_test['category']
y_train_funded = y_train['funded']
y_test_funded = y_test['funded']



####	TUNING 	####

#	Initial attempt to idenfiy optimal feature number for count vectorizer, and topic number for LDA
#	Method is to bootstrap predictive accuracy on another split of the training set
#	Algorithms set to default values are used, with tuning parameter changing each iteration

#	Resulting accuracy is plotted and ideal parameter ranged is obtained qualitatively
#	Generally looking for an 'elbow' in the plot, indicating reduction in marginal benefit
#	Computing power is also in consideration when choosing parameter ranges


#	Searching for ideal max words to vectorizer from the text fields

scores = []
for i in range(5):
    X_trn, X_tst, y_trn, y_tst = train_test_split(X_train_text, y_train_category, test_size=0.2, stratify=y_train)
    
    x = []
    for j in [1000,2000,4000,8000,12000,16000,20000]:
        
        classifier_test = Pipeline([
            ('vectorizer', CountVectorizer(max_features=j, analyzer='word', stop_words=StopWords)),
            ('tfidf', TfidfTransformer()),
            ('clf', LinearSVC(multi_class='ovr'))]) 
    
        classifier_test.fit(X_trn, y_trn)
        x.append(classifier_test.score(X_tst, y_tst))
        print(i,j)
    
    scores.append(x)
    
feat_scores = pd.DataFrame(scores, columns=[1000,2000,4000,8000,12000,16000,20000])
plt.plot(feat_scores.mean())

feat_scores.to_csv(path + 'feat_scores.csv')
n_feat = 8000


#	Searching for ideal number of topics to generate using LDA, assuming rougly estimated max features

#   ** This method uses predictive accuracy of default SVM algorithm to tune
#	** Ideal would be to use perplexity, a measure of topic generalizability, to tune, but it seems to be broken in sklearn

scores = []
for i in range(5):
    X_trn, X_tst, y_trn, y_tst = train_test_split(X_train_text, y_train_category, test_size=0.2, stratify=y_train)

    x = []
    for j in [10,20,30,50,100,200]:
        
        classifier_test = Pipeline([
            ('vectorizer', CountVectorizer(max_features=n_feat, analyzer='word', stop_words=StopWords)),
            ('lda', LatentDirichletAllocation(n_topics=j, n_jobs=-1)),
            ('clf', LinearSVC(multi_class='ovr'))])
    
        classifier_test.fit(X_trn, y_trn)
        x.append(classifier_test.score(X_tst, y_tst))
        print(i,j)
        
    scores.append(x)
    
topic_scores = pd.DataFrame(scores, columns=[10,20,30,50,100,200])
plt.plot(topic_scores.mean())

topic_scores.to_csv(path + 'topic_scores.csv')
n_topic = 50


#	Initial parameter ranges for transformer components
feat_rng = range(4000,12001,1000)
topic_rng = range(20, 61, 10)



#   Tuning the transforming steps using full gridsearch algorithm with default SVM classifier

classifier = Pipeline([
        ('vectorizer', CountVectorizer(analyzer='word', stop_words=StopWords)),
        ('lda', LatentDirichletAllocation(n_jobs=-1)),
        ('clf', LinearSVC(penalty='l1', dual=False, multi_class='ovr'))])

params = {'vectorizer__max_features':feat_rng,
          'lda__n_topics':topic_rng}

gs = GridSearchCV(classifier, param_grid=params, cv=5, n_jobs=-1)
gs.fit(X_train_text, y_train_category)

print(gs.best_score_)
print(gs.best_params_)
# Best: score 0.65  features 4000  topics 40


#   More granular tuning in the area of previous best feature and topic values

classifier = Pipeline([
        ('vectorizer', CountVectorizer(analyzer='word', stop_words=StopWords)),
        ('lda', LatentDirichletAllocation(n_jobs=-1)),
        ('clf', LinearSVC(penalty='l1', dual=False, multi_class='ovr'))])

params = {'vectorizer__max_features':[3000,3500,4000,4500,5000,5500],
          'lda__n_topics':range(35,51,2)}

gs = GridSearchCV(classifier, param_grid=params, cv=5, n_jobs=-1)
gs.fit(X_train_text, y_train_category)

print(gs.best_score_)
print(gs.best_params_)
# Best: score 0.67  features 3500  topics 47



#   Extracting components from best estimator

cv = gs.best_estimator_.named_steps['vectorizer']
ld = gs.best_estimator_.named_steps['lda']  
sv = gs.best_estimator_.named_steps['clf'] 


#   Identifying top words in each topic as well as mapping best topics to each category class

#	Top 30 words in each topic
topic_dict = {}
for i, j in enumerate(ld.components_):
    topic_dict[i] = ' '.join([cv.get_feature_names()[k] for k in j.argsort()[:-30:-1]])

topics = pd.DataFrame(topic_dict, index=['Top Words']).T
topics.to_csv(path + 'initial_topics.csv')


#	Mapping topic coefficient vectors to their respective category classes
cat_dict = {}
for i in range(len(sv.classes_)):
    cats[sv.classes_[i]] = sv.coef_[i]
categories = pd.DataFrame(cats)


#	Mapping the most positively and negatively predictive topics to each category class

pos = {}
neg = {}
for i in sv.classes_:
    pos[i] = categories[i].nlargest(5).index
    neg[i] = categories[i].nsmallest(5).index

influential_topics = pd.concat([pd.DataFrame(pos, index=range(1,6)), pd.DataFrame(neg, index=range(-1,-6,-1))])
influential_topics.to_csv(path + 'influential_topics.csv')



####	FINAL TUNING/MODELING 	####

#   Create final training set using LDA topic vectors and previous non-text columns

X_train_text_vectors = pd.DataFrame(ld.transform(cv.transform(X_train_text)),
                                    index=X_train_text.index, columns=['Topic {}'.format(i) for i in range(gs.best_params_['lda__n_topics'])]) 

X_train_final = pd.concat([X_train.drop('full_text', axis=1), X_train_text_vectors], axis=1)


#   Tuning SVM classifier on all prospective features

svm_params = {'C':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}

svm_gs = GridSearchCV(LinearSVC(penalty='l1', dual=False, multi_class='ovr'), param_grid=svm_params, cv=5, n_jobs=-1)
svm_gs.fit(X_train_final, y_train_category)

print(svm_gs.best_score_)
print(svm_gs.best_estimator_)
# Best: score 0.691  C 10000


#	More granular tuning

svm_params = {'C':range(1000,15001,1000)}

svm_gs = GridSearchCV(LinearSVC(penalty='l1', dual=False, multi_class='ovr'), param_grid=svm_params, cv=5, n_jobs=-1)
svm_gs.fit(X_train_final, y_train_category)

print(svm_gs.best_score_)
print(svm_gs.best_estimator_)
# Best: score 0.691  C 5000


#   Scoring model on test set for predictive accuracy
#	The test set text field must be transformed using count vectorizer and LDA generated in training

X_test_text_vectors = pd.DataFrame(ld.transform(cv.transform(X_test_text)),
                                   index=X_test_text.index, columns=['Topic {}'.format(i) for i in range(gs.best_params_['lda__n_topics'])])

X_test_final = pd.concat([X_test.drop('full_text', axis=1), X_test_text_vectors], axis=1)

print(svm_gs.best_estimator_.score(X_test_final, y_test_category))
# Testing accuracy 0.689



####	PERFORMANCE ASSESSMENT 	####

#	Generate predictions and classification metrics

pred = svm_gs.best_estimator_.predict(X_test_final)
print(classification_report(y_test_category, pred))

sns.heatmap(confusion_matrix(y_test_category,pred), annot=True)

conf_matrix = pd.DataFrame(confusion_matrix(y_test_category,pred),
                           index = svm_gs.best_estimator_.classes_,
                           columns = svm_gs.best_estimator_.classes_).T
conf_matrix.to_csv(path + 'conf_matrix.csv')


#   Plot learning curve to check for overfitting

train_scores = []
test_scores = []
for i in range(500,len(X_train_final),500):
    svm_gs.best_estimator_.fit(X_train_final[:i], y_train_category[:i])
    train_scores.append(svm_gs.best_estimator_.score(X_train_final[:i], y_train_category[:i]))

    test_scores.append(svm_gs.best_estimator_.score(X_test_final, y_test_category))


plt.plot(range(500,len(X_train_final),500), train_scores,)
plt.plot(range(500,len(X_train_final),500), test_scores)
#	Training and test curved both converge on final score

pd.DataFrame([train_scores, test_scores],
             columns=range(500,len(X_train_final),500),
             index=['train', 'test']).T.to_csv(path + 'learning_curve.csv')



#   Brief look at what topics might negatively impact funding success
#   It wouldn't do to recommend focusing on a topic if it's going to negatively impact funding

svm_fund = GridSearchCV(LinearSVC(penalty='l1', dual=False, multi_class='ovr'), param_grid=svm_params, cv=5, n_jobs=-1)
svm_fund.fit(X_train_text_vectors, y_train_funded)

print(svm_fund.best_score_)
#	Testing accuracy 0.738


#	Predictive accuracy is high for topics on funding success
#	Results are inconclusive regarding successful/unsuccessful topic themes
pd.DataFrame(svm_fund.best_estimator_.coef_[0], columns=['Funding Impact']).to_csv(path + 'topic_funding_impact.csv')
print(svm_fund.best_estimator_.score(X_test_text_vectors, y_test_funded))



#	Perplexity tuning
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





