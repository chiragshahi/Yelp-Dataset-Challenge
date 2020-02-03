from sklearn.feature_selection import f_classif
import numpy as np
import pandas as pd

# ['review_stars', 'review_funny_upvotes', 'review_useful_upvotes', 'review_cool_upvotes', 'total_tokens', 'compound_score_review', 'user_avg_stars', 'user_yelping_since', 'user_review_count']

df = pd.read_csv('/Users/chiragshahi/Desktop/final_Dataset.csv', header = 0)

X = df.values[:, 2:]
y = df.values[:, 1]

y = y.astype('int')

feature_scores = f_classif(X, y)[0]
print(feature_scores)