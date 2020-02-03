import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd 
import numpy as np 

df = pd.read_csv(
    './finalDatsetv3.csv',
    header=0,
    names = ['review_id', 'review_stars', 'review_funny_upvotes', 'review_useful_upvotes', 'review_cool_upvotes', 'total_tokens', 'compound_score_review', 'noun_count', 'pos_noun_count', 'neg_noun_count', 'neutral_noun_count', 'adverb_count', 'pos_adverb_count', 'neg_adverb_count', 'neutral_adverb_count', 'verb_count', 'pos_verb_count', 'neg_verb_count', 'neutral_verb_count', 'adjective_count', 'pos_adjective_count', 'neg_adjective_count', 'neutral_adjective_count', 'tot_pos_words_count', 'tot_neg_words_count', 'tot_neu_words_count', 'user_avg_stars', 'user_yelping_since', 'user_review_count'], 
    usecols = ['review_stars', 'review_useful_upvotes', 'review_cool_upvotes',
             'total_tokens', 'compound_score_review', 'noun_count', 
             'verb_count', 'adjective_count', 'tot_pos_words_count', 
             'tot_neg_words_count', 'tot_neu_words_count', 'user_avg_stars', 'user_yelping_since', 'user_review_count']
)

df = df[df.review_stars != 3]

X = df.values[:, 1:]
y = df.values[:, 0]

# Binarize the output
y = label_binarize(y, classes=[1, 2, 4, 5])
n_classes = y.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)

classifier = OneVsRestClassifier(KNeighborsClassifier(algorithm= 'auto', n_neighbors= 5, p= 1, weights= 'distance'))

y_score = classifier.fit(X_train, y_train).predict_proba(X_test) #.decision_function(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
colors = ['blue', 'red', 'green', 'yellow']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class data')
plt.legend(loc="lower right")
plt.show()