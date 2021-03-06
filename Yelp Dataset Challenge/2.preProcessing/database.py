import MySQLdb

config = {
	"host" : "localhost",
	#provide username to connect to sql
	"user" : "root",
	#provide password to connect to sql
	"passwd" : "p@ssword",
	"db" : "yelpdatasetfeatures"
};

class Database():
	#self = this;
	connection = None;
	cur = None;
	def establishConnection(self):
		self.connection = MySQLdb.connect(**config);
		self.cur = self.connection.cursor(); 
	def execute(self, query, params):
		self.cur.execute(query, params);
		self.connection.commit();
		

db = Database();
db.establishConnection();


'''
CREATE TABLE  yelpdatasetfeatures.features (
	review_id VARCHAR(100),
	review_stars INT,
	review_funny_upvotes INT,
	review_useful_upvotes INT, 
	review_cool_upvotes INT,
	total_tokens INT,
	compound_score_review FLOAT,
    noun_count INT, 
    pos_noun_count INT, 
    neg_noun_count INT, 
    neutral_noun_count INT, 
    adverb_count INT, 
    pos_adverb_count INT, 
    neg_adverb_count INT, 
    neutral_adverb_count INT, 
    verb_count INT, 
    pos_verb_count INT, 
    neg_verb_count INT, 
    neutral_verb_count INT, 
    adjective_count INT, 
    pos_adjective_count INT, 
    neg_adjective_count INT, 
    neutral_adjective_count INT,
    tot_pos_words_count INT,
    tot_neg_words_count INT,
    tot_neu_words_count INT,
	user_avg_stars FLOAT,
	user_yelping_since INT,
	user_review_count INT,
	PRIMARY KEY (review_id)
);
'''