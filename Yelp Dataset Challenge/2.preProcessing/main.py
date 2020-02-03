import pandas as pd;
from calculateNumOfDays import calculateNumOfDays;
from extractFeaturesFromReviews import extractFeaturesFromReviews;
from database import db
def groupReviewStars(star):
	if(star>3):
		return 1
	else:
		return 0

df = pd.read_csv(
	#name of file from which data has to be read
	'finalData.csv',
	usecols= [
		'review_id',
		'user_review_count',
		'yelping_since',
		'user_avg_stars',
		'user_avg_stars',
		'review_stars',
		'text',
		'useful',
		'funny',
		'cool'
	],
	encoding="utf-8");
featureDict = {};
for index, row in df.iterrows():
	#if index<5: 
		if int(row['review_stars']) == 3:
			continue;
		featureDict['review_id'] = row['review_id'];
		featureDict['user_review_count'] = int(row['user_review_count']);
		featureDict['user_yelping_since'] = calculateNumOfDays(row['yelping_since']);
		featureDict['user_avg_stars'] = row['user_avg_stars'];
		featureDict['review_stars'] = int(row['review_stars']);
		arr = extractFeaturesFromReviews(row['text']);
		featureDict['total_tokens'] = arr[0];
		featureDict['compound_score_review'] = arr[1];
		featureDict['noun'] = arr[2];
		featureDict['pos_noun'] = arr[3];
		featureDict['neg_noun'] = arr[4];
		featureDict['neutral_noun'] = arr[5];
		featureDict['adverb'] = arr[6];
		featureDict['pos_adverb'] = arr[7];
		featureDict['neg_adverb'] = arr[8];
		featureDict['neutral_adverb'] = arr[9];
		featureDict['verb'] = arr[10];
		featureDict['pos_verb'] = arr[11];
		featureDict['neg_verb'] = arr[12];
		featureDict['neutral_verb'] = arr[13];
		featureDict['adjective'] = arr[14];
		featureDict['pos_adjective'] = arr[15];
		featureDict['neg_adjective'] = arr[16];
		featureDict['neutral_adjective'] = arr[17];
		featureDict['tot_pos_words'] = arr[18],
		featureDict['tot_neg_words'] = arr[19],
		featureDict['tot_neu_words'] = arr[20],
		featureDict['review_useful_upvotes'] = int(row['useful']);
		featureDict['review_funny_upvotes'] = int(row['funny']);
		featureDict['review_cool_upvotes'] = int(row['cool']);
		print (index);
		db.execute(
			"INSERT INTO features (review_id, user_review_count, user_yelping_since, user_avg_stars, review_stars, total_tokens, compound_score_review, noun_count, pos_noun_count, neg_noun_count, neutral_noun_count, adverb_count, pos_adverb_count, neg_adverb_count, neutral_adverb_count, verb_count, pos_verb_count, neg_verb_count, neutral_verb_count, adjective_count, pos_adjective_count, neg_adjective_count, neutral_adjective_count, tot_pos_words_count, tot_neg_words_count, tot_neu_words_count, review_useful_upvotes, review_funny_upvotes, review_cool_upvotes) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", 
			(
				featureDict['review_id'],
				featureDict['user_review_count'],
				featureDict['user_yelping_since'],
				featureDict['user_avg_stars'],
				featureDict['review_stars'],
				featureDict['total_tokens'],
				featureDict['compound_score_review'],
				featureDict['noun'],
				featureDict['pos_noun'],
				featureDict['neg_noun'],
				featureDict['neutral_noun'],
				featureDict['adverb'],
				featureDict['pos_adverb'],
				featureDict['neg_adverb'],
				featureDict['neutral_adverb'],
				featureDict['verb'],
				featureDict['pos_verb'],
				featureDict['neg_verb'],
				featureDict['neutral_verb'],
				featureDict['adjective'],
				featureDict['pos_adjective'],
				featureDict['neg_adjective'],
				featureDict['neutral_adjective'],
				featureDict['tot_pos_words'],
				featureDict['tot_neg_words'],
				featureDict['tot_neu_words'],
				featureDict['review_useful_upvotes'],
				featureDict['review_funny_upvotes'],
				featureDict['review_cool_upvotes'],
			));
		#print(row['text']);
		#print (featureDict);
	#else:
	#	break
