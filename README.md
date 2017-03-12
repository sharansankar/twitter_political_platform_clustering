# twitter_political_platform_clustering
##Summary 
Clustering tweets on Key Issues from the Presidential Candidates of the 2016 US Election using NLP and ML

##Motivation 
Given this past US presidential election, it was apparent that voters were not always aware of the candidate's position on key issues. Thus voter's insights on candidates had been fragmented by the partial availability of data. This project aims to cluster the tweets of the presidential candidates of 2016 US Elections into clusters which represent a key issue. By doing so we can use machine learning to provide users with candidates' views on key issues. 

##Related Work 
The clustering method follows an exemplar-based approach based on the paper "Exemplar-Based Topic Detect in Twitter Streams" by Ahmed Elbagoury and Rania Ibrahim (http://www.aaai.org/ocs/index.php/ICWSM/ICWSM15/paper/viewFile/10533/104540). In this paper, the method introduced extracts tweets based on their variance from their similarity with other tweets. This is based on the grounds that a tweet's similarity to a group of tweets will fall under three cases: 
  1. The tweet is ubiquitous to the group. It will have low variance. 
  2. The tweet is similar to only a certain group of tweets, giving it a high variance. 
  3. The tweet is not similar to any tweets in the group. It will also have a low variance.
Thus from these 3 cases, the tweets that fall under case 2 can be said to be exemplar tweets. These tweets can then act as our centroids of a topic cluster. 

##Implementation 
###Tweets 
This clustering algorithm was implemented using the Hillary Clinton and Donald Trump Tweets dataset on kaggle (https://www.kaggle.com/benhamner/clinton-trump-tweets). 


##Data Exploration and Preprocessing
Intuitively, I would believe that the candidates would use twitter as a medium to explain key elements of their platform to the public. To analyze the data I thought by filtering the tweets and creating a word cloud of each candidates tweets, there would be key words that would appear ('immigration','jobs','military',etc.). I preprocessed the data by: 
  1. Tokenizing the tweets
  2. Filtering out punctuation and twitter handles
  3. lemmatizing all tokens 
  4. Filtering out stopwords 

After preprocessing the tweets, the wordclouds generated failed to abide by my assumptions: 
__Donald Trump Tweets__
![Trump](images/trump_cloud.png)

__Hillary Clinton Tweets__
![Hillary](images/clinton_cloud.png) 

As we can see the most frequent terms that candidates tweeted were their rivals. What stood out to me most was that Hillary's tweets talked most about Tump, but Trump's tweets also talked most about himself. Thus after looking at the tweets, I was unsure of the success of clustering by topic but decided to continue pursuing the goal.

d

