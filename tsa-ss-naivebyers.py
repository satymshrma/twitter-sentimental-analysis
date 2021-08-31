#%%
"""
Twitter Sentiment Analysis

Fetching data from twitter using tweepy and using textblob to get a sentimental analysis.

Mini Project by Satyam Sharma, B.Tech(CS/ML), 4rd sem-ML, Roll no.25
U.Roll.No:2013593

Dataset from : https://www.kaggle.com/c/twitter-sentiment-analysis2/data

Algo: Naive Bayes

"""


import re 
import tweepy 
import os
import pandas as pd
import warnings
import sklearn
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer #to convert strings into floats for the algo.
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from nltk.corpus import stopwords
from tweepy import OAuthHandler


DeprecationWarning('ignore')
warnings.filterwarnings('ignore')

#os.chdir('F:/Twitter Analysis')
df=pd.read_csv('train.csv')
vectorizer=TfidfVectorizer(use_idf=True,lowercase=True,strip_accents='ascii')
x=vectorizer.fit_transform(df['SentimentText'])
y=df['Sentiment']
log_model=naive_bayes.MultinomialNB()
log_model.fit(x,y)
score=log_model.score(x,y)
print("Model score against itself",score)
#test=pd.read_csv('test.csv')
#%%



class TwitterClient(object): 
	''' 
	Generic Twitter Class for sentiment analysis. 
	'''
	def __init__(self): 
		''' 
		Class constructor or initialization method. 
		'''
		# keys and tokens from the Twitter Dev Console 
		consumer_key = 'xxxx'
		consumer_secret = 'xxxx'
		access_token = 'xxxx'
		access_token_secret = 'xxxx'

		# attempt authentication 
		try: 
			# create OAuthHandler object 
			self.auth = OAuthHandler(consumer_key, consumer_secret) 
			# set access token and secret 
			self.auth.set_access_token(access_token, access_token_secret) 
			# create tweepy API object to fetch tweets 
			self.api = tweepy.API(self.auth) 
		except: 
			print("Error: Authentication Failed") 

	def clean_tweet(self, tweet): 
		''' 
		Utility function to clean tweet text by removing links, special characters 
		using simple regex statements. 
		'''
		return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split()) 

	def get_tweet_sentiment(self, tweet): 
		''' 
		Utility function to classify sentiment of passed tweet 
		using textblob's sentiment method 
		'''
		cleantweet=self.clean_tweet(tweet)
		cleantweet=[cleantweet] 
		sortedtweet=vectorizer.transform(cleantweet)
		
		# set sentiment 
		if log_model.predict(sortedtweet) == 0: 
			return 'negative'
		elif log_model.predict(sortedtweet) == 1: 
			return 'positive'

	def get_tweets(self, query, count = 10): 
		''' 
		Main function to fetch tweets and parse them. 
		'''
		# empty list to store parsed tweets 
		tweets = [] 

		try: 
			# call twitter api to fetch tweets 
			fetched_tweets = self.api.search(q = query, count = count)
			# parsing tweets one by one 
			for tweet in fetched_tweets: 
				# empty dictionary to store required params of a tweet 
				parsed_tweet = {} 

				# saving text of tweet 
				parsed_tweet['text'] = tweet.text 
				# saving sentiment of tweet 
				parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text) 

				# appending parsed tweet to tweets list 
				if tweet.retweet_count > 0: 
					# if tweet has retweets, ensure that it is appended only once 
					if parsed_tweet not in tweets: 
						tweets.append(parsed_tweet) 
				else: 
					tweets.append(parsed_tweet) 

			# return parsed tweets 
			return tweets 

		except tweepy.TweepError as e: 
			# print error (if any) 
			print("Error : " + str(e)) 

def main(): 
	# creating object of TwitterClient Class 
	api = TwitterClient() 
	# calling function to get tweets 
	tweets = api.get_tweets(query = input("Enter Query >"), count = 1200) 

	# picking positive tweets from tweets 
	ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive'] 
	# percentage of positive tweets 
	print("Positive tweets percentage: {} %".format(100*len(ptweets)/len(tweets))) 
	# picking negative tweets from tweets 
	ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative'] 
	# percentage of negative tweets 
	print("Negative tweets percentage: {} %".format(100*len(ntweets)/len(tweets))) 
	 
    
	# printing first 5 positive tweets 
	print("\n\nPositive tweets:") 
	for tweet in ptweets[:10]: 
		print(tweet['text']) 

	# printing first 5 negative tweets 
	print("\n\nNegative tweets:") 
	for tweet in ntweets[:10]: 
		print(tweet['text']) 

if __name__ == "__main__": 
	# calling main function 
	main() 





# %%
