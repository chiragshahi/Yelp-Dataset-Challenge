from sklearn.feature_selection import RFE
from sklearn.svm import SVR
import pandas as pd

df = pd.read_csv('/Users/chiragshahi/Desktop/final_Dataset.csv', header = 0)

X = df.values[:, 2:]
y = df.values[:, 1]

estimator = SVR(kernel="linear")
selector = RFE(estimator, step=1)
selector = selector.fit(X, y)

print(selector.support_)
print(selector.ranking_)
