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
