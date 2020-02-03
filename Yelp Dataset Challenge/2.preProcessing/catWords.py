import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords 
from collections import Counter

tokenizer = RegexpTokenizer(r'\w+')

def categoriseWords(review, category):
	stop_words = set(stopwords.words('english'))
	text = tokenizer.tokenize(review.lower())
	filtered_sentence = [w for w in text if not w in stop_words]
	tagged = nltk.pos_tag(filtered_sentence)
	counts = Counter(tag for word, tag in tagged)

	# normalized values
	# total = sum(counts.values())
	# return dict((word, float(count)/total) for word,count in counts.items())
	
	return counts[category]

print(categoriseWords("And now for something completely different working moving", "VBG"))