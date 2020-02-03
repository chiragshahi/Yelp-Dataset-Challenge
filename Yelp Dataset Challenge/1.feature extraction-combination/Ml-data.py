from pyspark.sql import SparkSession



spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()


#business
business = spark.read.json("yelp_academic_dataset_business.json")
business1=business.select((business['name'].alias('b_name')), (business['business_id'].alias('b_id')) ,(business['review_count'].alias('b_review_count')),business['attributes'],business['categories'],(business['stars'].alias('business_stars')))
business1.createOrReplaceTempView("business1")

#users
users = spark.read.json("yelp_academic_dataset_user.json")
users1=users.select((users['user_id'].alias('userid')), (users['average_stars'].alias('user_avg_stars')),users['name'],(users['review_count'].alias('user_review_count')),users['useful'],users['yelping_since'],users['funny'],users['cool'])
users1.createOrReplaceTempView("users1")


#reviews
reviews = spark.read.json("yelp_academic_dataset_review.json")
reviews1=reviews.select(reviews['user_id'], reviews['review_id'],reviews['business_id'],(reviews['stars'].alias('review_stars')),reviews['text'])
r1=reviews1.filter(reviews['stars'] == 1).limit(20000)
r2=reviews1.filter(reviews['stars'] == 2).limit(20000)
r3=reviews1.filter(reviews['stars'] == 3).limit(20000)
r4=reviews1.filter(reviews['stars'] == 4).limit(20000)
r5=reviews1.filter(reviews['stars'] == 5).limit(20000)
review=r1.union(r2)
review=review.union(r3)
review=review.union(r4)
review=review.union(r5)
review.createOrReplaceTempView("review")

#join
sqlDF = spark.sql("SELECT * from business1,review where business1.b_id==review.business_id ")
sqlDF.createOrReplaceTempView("sqlDF")
sql1= spark.sql("SELECT sqlDF.review_id, sqlDF.business_stars, sqlDF.user_id, users1.user_review_count, users1.yelping_since, users1.user_avg_stars, sqlDF.business_id, sqlDF.review_stars, sqlDF.text, users1.useful, users1.funny, users1.cool FROM sqlDF,users1 where users1.userid==sqlDF.user_id  ")



sql1.toPandas().to_csv("Ml-Proj-Data.csv", header=True)






