import pandas as pd 
from collections import Counter

df = pd.read_csv('ML-Proj-Data copy.csv')
df.insert(df.shape[1]+1, 'bcount', 0)

counts = Counter(df['user_id'])

for i in range(df.shape[0]):
	# df['bcount'][i] == counts[df['user_id'][i]]
	df.bcount.loc[i] = counts[df['user_id'][i]]

print(df)