from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
import pandas as pd

# ['review_stars', 'review_funny_upvotes', 'review_useful_upvotes', 'review_cool_upvotes', 'total_tokens', 'compound_score_review', 'user_avg_stars', 'user_yelping_since', 'user_review_count']

df = pd.read_csv('/Users/chiragshahi/Desktop/final_Dataset.csv', header = 0)


X = df.values[:, 2:]
y = df.values[:, 1]

y = y.astype('int')

pstv_X = X[:,X.min(axis=0)>=0]

X_new = SelectKBest(chi2, k=4).fit_transform(pstv_X, y)

print(X_new)