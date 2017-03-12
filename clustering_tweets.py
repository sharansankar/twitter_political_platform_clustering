import pandas as pd
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import SpectralClustering
import numpy as np
import math
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.stem.wordnet import WordNetLemmatizer

stop_words = stopwords.words('english') #+ ['donald', 'hillary', 'trump', 'clinton', 'ha', "trump's", 'thank', 'great','wa', 'new']
def read_data(path):
    df = pd.read_csv(path,header=0)
    df = df[['handle','text']]
    trump_tweets = df[df.handle == 'realDonaldTrump']
    trump_tweets = trump_tweets['text']
    clinton_tweets = df[df.handle == 'HillaryClinton']
    clinton_tweets = clinton_tweets['text']
    return trump_tweets.values, clinton_tweets.values

def clean_data(intweets):
    for x in range(len(intweets)):
        intweets[x] = intweets[x].decode('unicode_escape').encode('ascii','ignore')
        intweets[x] = intweets[x].translate(None,string.digits)
    tokenize_tweet = TweetTokenizer(preserve_case=False, strip_handles=True)
    tweets = []


    #tokenize tweets
    for x in range(len(intweets)):
        #removing hashtags
        intweets[x] = intweets[x].replace('#','')


        #remove tweets that have less than 3 words
        if len(tokenize_tweet.tokenize(intweets[x])) > 3:
            tweets.append(tokenize_tweet.tokenize(intweets[x]))

            # removing punctuation
            tweets[-1] = filter(lambda x: x not in string.punctuation, tweets[-1])


            #lemmatizing
            lemmatizer = WordNetLemmatizer()
            for y in range(len(tweets[-1])):
                #print tweets[-1][y]
                tweets[-1][y] = lemmatizer.lemmatize(tweets[-1][y])
                #print tweets[-1][y]

            #removing stop words
            tweets[-1] = filter(lambda x: x not in stop_words, tweets[-1])



            #remove links
            tweets[-1] = filter(lambda x: 'https' not in x, tweets[-1])
            tweets[-1] = ' '.join(tweets[-1])


    return tweets

def generate_similarity_matrix(tweets):
    vectorizer = TfidfVectorizer(lowercase=False)
    #print len(tweets)
    tfidf_matrix = vectorizer.fit_transform(tweets)
    sim_matrix = cosine_similarity(tfidf_matrix.transpose().todense())
    return sim_matrix,tfidf_matrix

def exemplar_tweet_extraction(sim_matrix, tweets, unfiltered_tweets):
    n = len(sim_matrix[1])

    #calculating variance
    mean_vec = [sum(sim_matrix[x,:])/n for x in range(n)]
    var_vec = []
    sorted_variances = []

    for x in range(n):
        buffer_val = 0
        for y in range(n):
            buffer_val += math.pow((sim_matrix[x,y]-mean_vec[x]),2)/(n-1)
        var_vec.append(buffer_val)
        sorted_variances.append(buffer_val)
    #performing exemplar tweet extraction algorithm
    topics = 0
    exemplar_tweet = []
    exemplar_index = []


    exemplar_tweet.append(tweets[var_vec.index(sorted_variances[0])])
    exemplar_tweet_index = var_vec.index(sorted_variances[0])
    current_tweet_index = exemplar_tweet_index

    exemplar_index.append(var_vec.index(sorted_variances[0]))

    while (topics < 9 ):
        #print sim_matrix[exemplar_tweet_index][current_tweet_index]
        exemplar_tweet_index = var_vec.index(sorted_variances[0])
        while sim_matrix[exemplar_tweet_index][current_tweet_index] > 0.01:
            #print sim_matrix[exemplar_tweet_index][current_tweet_index]
            sorted_variances.pop(0)
            exemplar_tweet_index = var_vec.index(sorted_variances[0])

        if tweets[exemplar_tweet_index] not in exemplar_tweet:
            exemplar_tweet.append(unfiltered_tweets[exemplar_tweet_index])
            exemplar_index.append(exemplar_tweet_index)
            topics += 1
            sorted_variances.pop(0)
            #print topics


    # for tweet in exemplar_tweet:
    #     print "------------tweet--------------"
    #     print tweet

    return exemplar_index

def exemplar_clustering(tweets,exemplar_indexes,sim_matrix):
    tweet_clusters = [[] for x in range(len(exemplar_indexes))]
    for x in range(len(tweets)):
        sim_vals = [0 for y in range(len(exemplar_indexes))]
        for y in range(len(exemplar_indexes)):
            sim_vals[y] = sim_matrix[x][y]

        if max(sim_vals) >= 0.10:
            tweet_clusters[sim_vals.index(max(sim_vals))].append(tweets[x])
    return tweet_clusters

def create_wordcloud(intweets):
    combined_tweets = ' '.join(intweets)


    word_cloud = WordCloud(font_path='/Library/Fonts/Verdana.ttf',
                          relative_scaling=1.0,
                          stopwords='None'
                          ).generate(combined_tweets)
    plt.imshow(word_cloud)
    plt.axis("off")
    plt.show()

def lda_clustering(tfidf):
    lda = LatentDirichletAllocation(n_jobs=-1)

    return lda.fit_transform(tfidf)

def spectral_clustering(tfidf, tweets):
    spectral = SpectralClustering(n_clusters=10)
    cluster = spectral.fit_predict(tfidf)
    #print cluster
    tweet_clusters = [[] for x in range(10)]
    for x in range(len(tweets)):
        tweet_clusters[cluster[x]].append(tweets[x])
    return tweet_clusters

if __name__ == '__main__':
    trump, clinton = read_data('tweets.csv')
    #clinton = np.concatenate((trump, clinton), axis=0)
    clinton_clean = clean_data(clinton)
    trump_clean = clean_data(trump)
    create_wordcloud(clinton_clean)

    # clinton_sim_matrix, clinton_tfidf = generate_similarity_matrix(clinton_clean)
    # exemplar_clinton = exemplar_tweet_extraction(clinton_sim_matrix,clinton_clean, clinton)
    #
    # clinton_clustered = exemplar_clustering(clinton,exemplar_clinton, clinton_sim_matrix)
    #
    # trump_clean = clean_data(trump)
    # trump_sim_matrix,trump_tfidf = generate_similarity_matrix(trump_clean)
    # exemplar_trump = exemplar_tweet_extraction(trump_sim_matrix.transpose(),trump_clean, trump)
    # trump_clustered = exemplar_clustering(trump,exemplar_trump,trump_sim_matrix)


    # for x in range(len(trump_clustered)):
    #     print "----------------cluster: ",x,"------------------"
    #     print trump_clustered[x]