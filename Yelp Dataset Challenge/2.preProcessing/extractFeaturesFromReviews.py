import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import sentiwordnet as swn

tokenizer = RegexpTokenizer(r'\w+')

sentimentIntensityAnalyzer = SentimentIntensityAnalyzer();

noun = ['NN', 'NNS', 'NNP', 'NNPS'];
adverb = ['RB', 'RBR', 'RBS', 'WRB'];
verb = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'];
adjective = ['JJ', 'JJS', 'JJR'];
def extractFeaturesFromReviews(review):
	stop_words = set(stopwords.words('english'));
	#Remove all stop words from a sentence.
	#Before removing the stop words make review lowercase.
	word_tokens = tokenizer.tokenize(review.lower());
	filtered_sentence = [w for w in word_tokens if not w in stop_words];
	#print(review)
	#print(filtered_sentence)
	#featureOne :  total number of words in a sentence
	numOfTokens = len(filtered_sentence);
	pos_count = {
		'noun' : 0,
		'pos_noun' : 0,
		'neg_noun' : 0,
		'neutral_noun' : 0, 
		'adjective' : 0,
		'pos_adjective' : 0,
		'neg_adjective' : 0,
		'neutral_adjective' : 0,
		'verb' : 0,
		'pos_verb' : 0,
		'neg_verb' : 0,
		'neutral_verb' : 0,
		'adverb' : 0,
		'pos_adverb' : 0,
		'neg_adverb' : 0,
		'neutral_adverb' : 0,
		'tot_pos_words': 0,
		'tot_neg_words': 0,
		'tot_neu_words': 0
	};

	pos = nltk.pos_tag(filtered_sentence);
	for tag in pos:
		if tag[1] in noun:
			pos_count['noun'] += 1;
			if(len(list(swn.senti_synsets(tag[0], 'n')))!=0):
				sentiment = list(swn.senti_synsets(tag[0], 'n'))[0];
				pos_score = sentiment.pos_score();
				neg_score = sentiment.neg_score();
				if pos_score == neg_score: 
					pos_count['neutral_noun'] += 1;
					pos_count['tot_neu_words'] += 1;
				elif pos_score > neg_score:
					pos_count['pos_noun'] += 1;
					pos_count['tot_pos_words'] += 1;
				else:
					pos_count['neg_noun'] += 1;
					pos_count['tot_neg_words'] += 1;
		elif tag[1] in adverb:
			pos_count['adverb'] += 1;
			if(len(list(swn.senti_synsets(tag[0], 'r')))!=0):
				sentiment = list(swn.senti_synsets(tag[0], 'r'))[0];
				pos_score = sentiment.pos_score();
				neg_score = sentiment.neg_score();
				if pos_score == neg_score: 
					pos_count['neutral_adverb'] += 1;
					pos_count['tot_neu_words'] += 1;
				elif pos_score > neg_score:
					pos_count['pos_adverb'] += 1;
					pos_count['tot_pos_words'] += 1;
				else:
					pos_count['neg_adverb'] += 1;
					pos_count['tot_neg_words'] += 1;
		elif tag[1] in verb:
			pos_count['verb'] += 1;
			if(len(list(swn.senti_synsets(tag[0], 'v')))!=0):
				sentiment = list(swn.senti_synsets(tag[0], 'v'))[0];
				pos_score = sentiment.pos_score();
				neg_score = sentiment.neg_score();
				if pos_score == neg_score: 
					pos_count['neutral_verb'] += 1;
					pos_count['tot_neu_words'] += 1;
				elif pos_score > neg_score:
					pos_count['pos_verb'] += 1;
					pos_count['tot_pos_words'] += 1;
				else:
					pos_count['neg_verb'] += 1;
					pos_count['tot_neg_words'] += 1;
		elif tag[1] in adjective:
			pos_count['adjective'] += 1;
			if(len(list(swn.senti_synsets(tag[0], 'a')))!=0):
				sentiment = list(swn.senti_synsets(tag[0], 'a'))[0];
				pos_score = sentiment.pos_score();
				neg_score = sentiment.neg_score();
				if pos_score == neg_score: 
					pos_count['neutral_adjective'] += 1;
					pos_count['tot_neu_words'] += 1;
				elif pos_score > neg_score:
					pos_count['pos_adjective'] += 1;
					pos_count['tot_pos_words'] += 1;
				else:
					pos_count['neg_adjective'] += 1;
					pos_count['tot_neg_words'] += 1;
	#print (pos_count);
	scores = sentimentIntensityAnalyzer.polarity_scores(review);
	#featureTwo : total score of the review; ranges between -1 to 1 
	compound_score = scores['compound'];
	#print (compound_score)
	return [
		numOfTokens,
		compound_score,
		pos_count['noun'],
		pos_count['pos_noun'],
		pos_count['neg_noun'],
		pos_count['neutral_noun'],
		pos_count['adverb'],
		pos_count['pos_adverb'],
		pos_count['neg_adverb'],
		pos_count['neutral_adverb'],
		pos_count['verb'],
		pos_count['pos_verb'],
		pos_count['neg_verb'],
		pos_count['neutral_verb'],
		pos_count['adjective'],
		pos_count['pos_adjective'],
		pos_count['neg_adjective'],
		pos_count['neutral_adjective'],
		pos_count['tot_pos_words'],
		pos_count['tot_neg_words'],
		pos_count['tot_neu_words']
	];

#extractFeaturesFromReviews("Antionette Morgan is an artist alright! A con artist! Alley kat sucks! I went in for one of their fake specials,And i paid the nominal fee . When I brought up a question about my design,the owner Antoinette Morgan blew up calling me a dumb bitch and then told me to leave. I said gladly, and that I wanted my refund. She called me an ugly bitch and told me to read the sign, which said no refunds! I didn't even recieve the tattoo! That place is evil and gives tattoo shops in that area a bad imag!"); 
