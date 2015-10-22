#!/usr/bin/python
'''
NLTK using movie reviews of Mad Max from various sources
'''

from twython import Twython
import json

def sentiment_score(text,pos_list,neg_list):
    pos_score=0
    neg_score=0
    for w in text.split(' '):
        if w in pos_list: pos_score+=1
        if w in neg_list: neg_score+=1
    return pos_score-neg_score

def sentiment(tweets,pos_words,neg_words):
    return [sentiment_score(tweet,positive_words,negative_words) for tweet in tweets]

def get_tweets(twython_object,query,n):
    count=0
    result_generator=twython_object.cursor(twython_object.search, q = query)
    result_set=[]
    for r in result_generator:
        result_set.append(r['text'])
        count+=1
        if count == n: break
    return result_set

if __name__ == "__main__":
    data={}
    data['bbc'] = open('../data/madmax_review/bbc.txt', 'r').read()
    data['forbes'] = open('../data/madmax_review/forbes.txt', 'r').read()
    data['guardian'] = open('../data/madmax_review/guardian.txt', 'r').read()
    data['moviepilot'] = open('../data/madmax_review/moviepilot.txt', 'r').read()

# regular expression
    import re

    from nltk.corpus import stopwords
# GUI for installing stopwords(if not already done)
# nltk.download_gui()

    stopwords_list=stopwords.words('english')
    stopwords_list=stopwords_list+['mad','max','film','fury','miller','road']

    for k in data.keys():
        # convert all text to lowercase
        data[k] = data[k].lower()
        # remove punctuation and replace with ' '
        # and remove numbers
        data[k] = re.sub(r'[-./?!,":;()\'|0-9]',' ',data[k])
        data[k] = data[k].split()
        # if not a stopword, add to list
        data[k] = [w for w in data[k] if not w in stopwords_list]

    print data['bbc'][:80]

# wordclouds
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    wordcloud=WordCloud(width=1000,height=500).generate(' '.join(data['bbc']))

    fig=plt.figure('bbc',figsize=(15,8))
    plt.imshow(wordcloud)
    plt.axis('off')

    wordcloud=WordCloud(width=1000,height=500).generate(' '.join(data['forbes']))

    fig=plt.figure('forbes',figsize=(15,8))
    plt.imshow(wordcloud)
    plt.axis('off')

    wordcloud=WordCloud(width=1000,height=500).generate(' '.join(data['guardian']))

    fig=plt.figure('guardian',figsize=(15,8))
    plt.imshow(wordcloud)
    plt.axis('off')

    wordcloud=WordCloud(width=1000,height=500).generate(' '.join(data['moviepilot']))

    fig=plt.figure('moviepilot',figsize=(15,8))
    plt.imshow(wordcloud)
    plt.axis('off')

    from nltk.tokenize import word_tokenize,sent_tokenize
# Tokenize a string to split off punctuation other than periods
    data_forbes = open('../data/madmax_review/forbes.txt', 'r').read()
    word_data = word_tokenize(data_forbes)
    print word_data[:24]
# tokenize by sentences
    print sent_tokenize(data_forbes)[:5]

    from nltk import pos_tag
# part-of-speech (pos) tagging - used for categorizing words
    pos_word_data = pos_tag(word_data)
    print pos_word_data[:10]
    from nltk import help
    print help.upenn_tagset('NNS')
    print help.upenn_tagset('NN')
    print help.upenn_tagset('IN')
    print help.upenn_tagset('TO')
    print help.upenn_tagset('DT')
    print help.upenn_tagset('CC')

# Stemming and lemmatization
    from nltk.stem.porter import PorterStemmer

# most common stemmer
    porter_stemmer=PorterStemmer()

    print 'Porter stemming'
    for w in word_data[:20]:
        print 'Actual: %s Stem %s' % (w,porter_stemmer.stem(w))

    from nltk.stem.snowball import SnowballStemmer
    snowball_stemmer=SnowballStemmer("english")

    print '\n'
    print 'Snowball stemming'
    for w in word_data[:20]:
        print 'Actual: %s Stem %s' % (w,snowball_stemmer.stem(w))

    from nltk.stem.lancaster import LancasterStemmer
# fastest stemming algorithm
    lancaster_stemmer = LancasterStemmer()
    print '\n'
    print 'Lancaster stemming'
    for w in word_data[:20]:
        print 'Actual: %s Stem %s' % (w,lancaster_stemmer.stem(w))

# Lemmatization
    from nltk.stem import WordNetLemmatizer
    wordnet_lemmatizer = WordNetLemmatizer()
    print '\n'
    print 'Lemmatization'
    for w in word_data[:20]:
        print 'Actual: %s Lemm %s' % (w,wordnet_lemmatizer.lemmatize(w))

# Stanford Named Entity Recognizer
# http://nlp.stanford.edu
    from nltk.tag.stanford import NERTagger
    print '\nPerforming NER tagging: '
    st = NERTagger('./stanford-ner-2014-06-16/classifiers/english.all.3class.distsim.crf.ser.gz','./stanford-ner-2014-06-16/stanford-ner.jar')
    print st.tag('''Barrack Obama is the president of the United States of America . His father is from Kenya and Mother from United States of America. He has two daughters with his wife. He has strong opposition in Congress due to Republicans'''.split())

#Please provide your keys here
    TWITTER_APP_KEY = 'XXXXXXXXXXXXXX'
    TWITTER_APP_KEY_SECRET = 'XXXXXXXXXXXXXX'
    TWITTER_ACCESS_TOKEN = 'XXXXXXXXXXXXXXXXXXXXXX'
    TWITTER_ACCESS_TOKEN_SECRET = 'XXXXXXXXXXXXXXXXXXXXX'

    t = Twython(app_key=TWITTER_APP_KEY,
                app_secret=TWITTER_APP_KEY_SECRET,
                oauth_token=TWITTER_ACCESS_TOKEN,
                oauth_token_secret=TWITTER_ACCESS_TOKEN_SECRET)

# get access to tweets
    with open('../data/politician_tweets.json') as fp:
        tweets=json.load(fp)

    # fetch tweets
    tweets={}
    max_tweets=300
    tweets['obama'] = [re.sub(r'[-.#/?!,":;()\']',' ',tweet.lower()) for tweet in get_tweets(t,'#obama',max_tweets)]
    tweets['hillary'] = [re.sub(r'[-.#/?!,":;()\']',' ',tweet.lower()) for tweet in get_tweets(t,'#hillary',max_tweets)]
    tweets['justintrudeau'] = [re.sub(r'[-.#/?!,":;()\']',' ',tweet.lower()) for tweet in get_tweets(t,'#justintrudeau',max_tweets)]
    tweets['putin'] = [re.sub(r'[-.#/?!,":;()\']',' ',tweet.lower()) for tweet in get_tweets(t,'#putin',max_tweets)]
    tweets['davidcameron'] = [re.sub(r'[-.#/?!,":;()\']',' ',tweet.lower()) for tweet in get_tweets(t,'#davidcameron',max_tweets)]
    tweets['donaldtrump'] = [re.sub(r'[-.#/?!,":;()\']',' ',tweet.lower()) for tweet in get_tweets(t,'#donaldtrump',max_tweets)]

    # perform sentiment analysis
    positive_words=open('../data/positive-words.txt').read().split('\n')
    negative_words=open('../data/negative-words.txt').read().split('\n')

    tweets_sentiment={}
    tweets_sentiment['obama'] = sentiment( tweets['obama'],positive_words,negative_words)
    tweets_sentiment['hillary'] = sentiment( tweets['hillary'],positive_words,negative_words)
    tweets_sentiment['justintrudeau'] = sentiment( tweets['justintrudeau'],positive_words,negative_words)
    tweets_sentiment['putin'] = sentiment( tweets['putin'],positive_words,negative_words)
    tweets_sentiment['davidcameron'] = sentiment( tweets['davidcameron'],positive_words,negative_words)
    tweets_sentiment['donaldtrump'] = sentiment( tweets['donaldtrump'],positive_words,negative_words)

    for entry in tweets_sentiment.keys():
        plt.figure(entry)
        plt.hist(tweets_sentiment[entry],10,facecolor='green',alpha=0.5)
        plt.xlabel('Sentiment Score')
        _ = plt.xlim([-4,4])
    plt.show()


