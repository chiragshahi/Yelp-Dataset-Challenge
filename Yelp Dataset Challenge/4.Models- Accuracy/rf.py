from __future__ import print_function

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

import pandas as pd 
import numpy as np 

df = pd.read_csv(
    'finalDatsetv3.csv',
    header=0,
    names = ['review_id', 'review_stars', 'review_funny_upvotes', 'review_useful_upvotes', 'review_cool_upvotes', 'total_tokens', 'compound_score_review', 'noun_count', 'pos_noun_count', 'neg_noun_count', 'neutral_noun_count', 'adverb_count', 'pos_adverb_count', 'neg_adverb_count', 'neutral_adverb_count', 'verb_count', 'pos_verb_count', 'neg_verb_count', 'neutral_verb_count', 'adjective_count', 'pos_adjective_count', 'neg_adjective_count', 'neutral_adjective_count', 'tot_pos_words_count', 'tot_neg_words_count', 'tot_neu_words_count', 'user_avg_stars', 'user_yelping_since', 'user_review_count'], 
    usecols = ['review_stars', 'review_useful_upvotes', 'review_cool_upvotes',
             'total_tokens', 'compound_score_review', 'noun_count', 
             
             'verb_count', 'adjective_count', 'tot_pos_words_count', 
             'tot_neg_words_count', 'tot_neu_words_count', 'user_avg_stars', 'user_yelping_since', 'user_review_count']
)

X = df.values[:, 1:]
y = df.values[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

tuned_parameters = [{'n_estimators': [50, 80, 100], 'max_depth': [90, 120, 150],
 'min_samples_split': [30, 50, 80], 'criterion': ['gini', 'entropy']}]

scores = ['accuracy']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5,
                       scoring='%s' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print("Detailed confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Accuracy Score: \n")
    print(accuracy_score(y_true, y_pred))

    print()