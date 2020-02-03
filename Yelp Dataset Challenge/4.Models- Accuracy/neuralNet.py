# This example has been taken from SciKit documentation and has been
# modifified to suit this assignment. You are free to make changes, but you
# need to perform the task asked in the lab assignment


from __future__ import print_function

#from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
print(__doc__)

# Loading the Digits dataset
#digits = datasets.load_digits()
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv(
    './finalData.csv',
    header=0,
    names = ['review_id', 'review_stars', 'review_funny_upvotes', 'review_useful_upvotes', 'review_cool_upvotes', 'total_tokens', 'compound_score_review', 'noun_count', 'pos_noun_count', 'neg_noun_count', 'neutral_noun_count', 'adverb_count', 'pos_adverb_count', 'neg_adverb_count', 'neutral_adverb_count', 'verb_count', 'pos_verb_count', 'neg_verb_count', 'neutral_verb_count', 'adjective_count', 'pos_adjective_count', 'neg_adjective_count', 'neutral_adjective_count', 'tot_pos_words_count', 'tot_neg_words_count', 'tot_neu_words_count', 'user_avg_stars', 'user_yelping_since', 'user_review_count'], 
    #usecols = ['review_stars', 'review_funny_upvotes', 'review_useful_upvotes', 'review_cool_upvotes', 'total_tokens', 'compound_score_review', 'noun_count', 'pos_noun_count', 'neg_noun_count', 'neutral_noun_count', 'adverb_count', 'pos_adverb_count', 'neg_adverb_count', 'neutral_adverb_count', 'verb_count', 'pos_verb_count', 'neg_verb_count', 'neutral_verb_count', 'adjective_count', 'pos_adjective_count', 'neg_adjective_count', 'neutral_adjective_count', 'tot_pos_words_count', 'tot_neg_words_count', 'tot_neu_words_count', 'user_avg_stars', 'user_yelping_since', 'user_review_count']
    usecols = [
        'review_stars',
        'review_useful_upvotes',
        'review_cool_upvotes',
        'total_tokens',
        'compound_score_review',
        'noun_count', 
        'verb_count',
        'adjective_count',
        'tot_pos_words_count', 
        'tot_neg_words_count',
        'tot_neu_words_count',
        'user_avg_stars',
        'user_yelping_since',
        'user_review_count'
    ]
); 

#print (df) 

df = df.astype('float')
df = df[df.review_stars != 3]
lst = df.values.tolist();

#lst.pop(0);
#print (lst);

#scaler = MinMaxScaler();
#arr = scaler.fit_transform(lst);
X = [];
y = [];
for ele in lst:
    X.append(ele[1:]);
    y.append(ele[0]);
#print (X);
#print (y);
#X = X.astype(float)

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
#n_samples = len(digits.images)
# = digits.images.reshape((n_samples, -1))
#y = digits.target

#X = lst[0:5]
#y = lst[6]


# Split the dataset in two equal parts into 80:20 ratio for train:test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# This is a key step where you define the parameters and their possible values
# that you would like to check.
#tuned_parameters = [{'kernel': ['rbf', 'linear'], 'gamma': [1e-3, 1e-4],
#                     'C': [1, 10, 100, 1000]}
#                    ]
tuned_parameters = [
    {   
        'hidden_layer_sizes' : [(10, 2), (20, 2), (2, 20)],  
        'activation' : ['identity', 'logistic', 'tanh', 'relu'],
        #'alpha' : [0.0001, 0.001, 0.01],
        'learning_rate' : ['constant', 'invscaling', 'adaptive'], 
        'max_iter' : [100, 10],
        'random_state': [5]

    }
]
# We are going to limit ourselves to accuracy score, other options can be
# seen here:
# http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
# Some other values used are the predcision_macro, recall_macro
scores = ['accuracy']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(MLPClassifier(), tuned_parameters, cv=5,
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

# Note the problem is too easy: the hyperparameter plateau is too flat and the
# output model is the same for precision and recall with ties in quality.
