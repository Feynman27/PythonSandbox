#!/usr/bin/python

import pandas as pd
import numpy as np
import gzip
import json
import nltk
import string
import os
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer,CountVectorizer
from nltk.stem.porter import PorterStemmer
import sys
from sklearn.externals import joblib
from sklearn import decomposition
import pickle
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import sentiment as vaderSentiment
from sklearn.decomposition import NMF, LatentDirichletAllocation
from scipy import stats
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot

def getPeriods(i):
  if i==0:
    return 500
  elif i==1:
    return 50
  else:
    return 50

def getPredictionWindow(i):
  if i == 0:
    return ('2014-07-01','2014-08-01')
  elif i == 1:
    return ('2014-07-01','2014-08-01')
  else :
    return ('2014-08-01','2014-08-01')

def objfunc(order):
  try:
    fit=sm.tsa.ARIMA(dta,order).fit()
    print 'aic:',fit.aic, ' for grid', order
    return fit.aic

  except:
    pass


def corpusToTxt(corpus,name):
  i=0
  for rev in corpus.values():
    _dir = 'reviews/'
    f = open(_dir+name+str(i)+'.txt','w+')
    f.write(rev)
    f.close()
    i+=1

def SaveDictionary(vocab,name):

   #This function saves the word dictonary built out of the abstracts so that it doesn't
   #have to be rebuilt each runtime.
#   pickle.dump(vocab,open("feature.pkl","wb"))
   with open(name+'.pkl','wb') as handle:
     pickle.dump(vocab,handle)


def SaveVectors(vecs,name):

   #This function saves the pre-built tf-idf vectors built out of the abstracts.
   #Use this in conjunction with the list of filenames and the saved vocabulary
   #To reload the model and make recommendations on new, unseen, documents.
   #from sklearn.externals import joblib
   #joblib.dump(vecs, 'tfidf.pkl')
   with open(name+'.pkl','wb') as handle:
     pickle.dump(vecs,handle)

def SaveTransform(tf, name):
  with open(name+'.pkl','wb') as handle:
    pickle.dump(tf,handle)

def SaveVectorizer(tfidf,name):
    #This function saves the vectorizer input, including the tfidf.idf_ vector
    #Which contains the idf weights needed to properly vectorize future documents.
    #Without it, new documents will only get approximately accurate vector representations
    #Upon reloading the model.
#    joblib.dump(tfidf, 'Vectorizer.pkl')
    with open(name+'.pkl','wb') as handle:
      pickle.dump(tfidf,handle)

## Find top competitors via cosine similarities
from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.metrics.pairwise import linear_kernel
def cosineSimilarity(target_description,product):
  print 'Calculating cosine similarity'
  ## create sparse matrix for target
  tfidf = joblib.load('description_Vectorizer_'+product+'.pkl')
  tfidf_matrix = joblib.load('tfidf_description_transform_'+product+'.pkl')
#  tfidf = joblib.load('description_Vectorizer.pkl')
#  tfidf_matrix = joblib.load('tfidf_description_transform.pkl')
  #vectorizer = CountVectorizer(tokenizer=tokenize,max_df=0.95,min_df=2,\
  #                                    stop_words='english', \
  #                                    ngram_range=(1,1))
  #target_matrix = vectorizer.transform(target_description)
  target_matrix = tfidf.transform([target_description])
  print target_matrix
  scores = cosine_similarity(target_matrix, tfidf_matrix)
  #scores = linear_kernel(target_matrix,tfidf_matrix)
  score_dict={}
  i=0
  for score in scores.flatten():
    score_dict[i] = score
    i+=1
  print 'done'
  return (score_dict)

## Fit tfidf using all product descriptions for this category
import collections
def competitorTfIdf(meta_dta, target_id, doFit=True):
  description_dict = collections.OrderedDict()
  product_id = collections.OrderedDict()
  competitor_urls = collections.OrderedDict()
  #target_categories = meta_dta['categories'][meta_dta['asin']==target_id][0]

  ## (asin,description) dict
  print 'Building competitor dictionaries....'
  print 'Number of items:', len(meta_dta['asin'])
  i=0
  ## Product index in df is 78635
  #for asin in meta_dta['asin'][78000:88001].tolist():
  for asin in meta_dta['asin'].tolist():
    if i%1000==0:
      print i
    if i==50000: break;
    #other_categories = meta_dta['categories'][meta_dta['asin']==asin][0]

    product_id[i] = asin
    url = str(meta_dta['imUrl'][meta_dta['asin']==asin].iloc[0])
    competitor_urls[i] = url
    description = str(meta_dta['description'][meta_dta['asin']==asin].iloc[0])
    lowers = description.lower()
    no_punctuation = lowers.translate(None,string.punctuation)
    description_dict[i] = no_punctuation
    i+=1

  print 'Done.'
  if doFit:
    print 'Performing TFIDF on product descriptions...'
    ## Train tfidf on all product descriptions for this category
    ## Create sparse matrix
    tfidf = TfidfVectorizer(tokenizer=tokenize,stop_words='english',\
        ngram_range=(1, 1),lowercase=True,strip_accents="unicode")

    ## Term-document matrix
    tfs = tfidf.fit_transform(description_dict.values())

    SaveTransform(tfs,'tfidf_description_transform_'+product)
    SaveVectorizer(tfidf,'description_Vectorizer_'+product)
    SaveVectors(tfidf.idf_,'tfidf_description_'+product)
    SaveDictionary(tfidf.vocabulary_,'feature_description_'+product)
    print 'done.'

  return (description_dict, product_id, competitor_urls)




def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

## Print top words in each LDA topic
def print_top_words(model, feature_names, n_top_words):
  for topic_idx, topic in enumerate(model.components_):
    print("Topic #%d:" % topic_idx)
    print(":".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
  print

def getDF(path):
  i = 0
  df = {}

  for d in parse(path):
    df[i] = d
    i +=1
#    if i == 500000: break ## hack for local testing

  #return pd.read_json(path,orient='index')
  return pd.DataFrame.from_dict(df,orient='index')

def stem_tokens(tokens,stemmer):
  stemmed = []
  for item in tokens:
    stemmed.append(stemmer.stem(item))
  return stemmed

## With stemming
def tokenize(text):
  tokens = nltk.word_tokenize(text)
  stems = stem_tokens(tokens,stemmer)
  return stems

## No stemming
def tokenize_nostem(text):
  tokens = nltk.word_tokenize(text)
  #stems = stem_tokens(tokens,stemmer)
  stems = tokens
  return stems

from nltk.tag import pos_tag
##strip any proper nouns (NNP) or plural proper nouns (NNPS) from a text
def strip_proppers_POS(text):
  tagged = pos_tag(text.split()) #use NLTK's part of speech tagger
  non_propernouns = [word for word,pos in tagged if pos != 'NNP' and pos != 'NNPS']
  return non_propernouns

#################################################################
'''
Pre-processing of input json file format --> csv format for
easier and quicker loading
'''
#################################################################
if __name__ == '__main__':
  ## Turn on to convert json-->csv
  doRightToCSV = False
  ## Turn on to do competitor analysis using tfidf
  ## with cosine similarity of product descriptions
  doCompetitors = False
  ## If off, will create csv for only 1 product id
  runFromProductCSV = True
  doTfidf = False
  #pathname ='./data/reviews_Health_and_Personal_Care.json.gz'
  #pathname_meta ='./data/meta_Health_and_Personal_Care.json.gz'
  #pathname ='./output.json'
  path_list=['./data/reviews_Health_and_Personal_Care.json.gz',\
             './data/reviews_Sports_and_Outdoors.json.gz',\
             './data/reviews_Home_and_Kitchen.json.gz',\
             './data/reviews_Electronics.json.gz',\
             './data/reviews_Clothing_Shoes_and_Jewelry.json.gz'
              ]
  meta_list=['./data/meta_Health_and_Personal_Care.json.gz',\
             './data/meta_Sports_and_Outdoors.json.gz',\
             './data/meta_Home_and_Kitchen.json.gz',\
             './data/meta_Electronics.json.gz',\
             './data/meta_Clothing_Shoes_and_Jewelry.json.gz'
            ]

  ## Category index
  cat_idx = int(raw_input('Enter category index (0-3): '))
  if (cat_idx>3):
    print 'Category index out of range.'
    sys.exit(0)

  print 'Running on category files:'
  print path_list[cat_idx]
  print meta_list[cat_idx]
  print

  max_product = ''
  product = ''
  product_list = ['B001KXZ808','B00BGO0Q9O','B001N07KUE','B0074BW614']

  ## Convert to csv and exit
  if doRightToCSV:
    pathname = path_list[cat_idx]
    pathname_meta = meta_list[cat_idx]
    print 'Converting json files to csv...'
    print pathname
    df = getDF(pathname)
    df.to_csv(pathname+'.csv')
    print pathname_meta
    df_meta = getDF(pathname_meta)
    df_meta.to_csv(pathname_meta+'.csv')
    print df.head()
    print df_meta.head()
    ## Uncomment to convert entire list, but
    ## memory is currently an issue
    #for pathname,pathname_meta in zip(path_list,meta_list):
    #  print pathname
    #  df = getDF(pathname)
    #  df.to_csv(pathname+'.csv')
    #  print pathname_meta
    #  df_meta = getDF(pathname_meta)
    #  df_meta.to_csv(pathname_meta+'.csv')
    #  print df.head()
    #  print df_meta.head()
    print 'done.'
    sys.exit(0)

  ## Read in meta data and merge to
  ## the dataframe on column=asin (i.e. product id)
  elif not runFromProductCSV:
    pathname = path_list[cat_idx]
    pathname_meta = meta_list[cat_idx]
    df = pd.read_csv(pathname + '.csv')
    df_meta = pd.read_csv(pathname_meta+ '.csv')
    df_meta['categories'] = \
        df_meta['categories'].map(lambda x: x.lstrip('[[').rstrip(']]').split(','))

    ## Read in only certain columsn from metadata
    df_meta = df_meta[['asin','description','title','imUrl','categories']]
    #df = getDF(pathname)

    ## Use only product with maximum reviews
    max_product = df['asin'].value_counts().index[0]
    df = df[df['asin']==str(max_product)].reset_index(drop=True)
    df = pd.merge(df, df_meta, on='asin', how='outer')

    if df.description[0] == str(np.nan):
      print 'WARNING: Description for this product is nan.'
    ## NaN values, so put in manually
    if cat_idx==1:
      df['description']='Rubber. Imported. Tracks steps, distance, calories burned and active minutes. Monitor how long and well you sleep. Wakes you (and not your partner) with a silent wake alarm. LED lights show how your day is stacking up against your goal. Slim, comfortable and easy to wear (sold with both large and small wristbands included with an interchangeable clasp). Sync stats wirelessly and automatically to your computer and over 150 leading smartphones'

      df_meta['description'][df_meta['asin']==max_product] = str(df['description'][0])
      #print df_meta['description'][df_meta['asin']==max_product].iloc[0]

    product = max_product
    print 'Writing csv for product: %s ' % product
    df.to_csv('./data/'+product+'.csv')

  ## Load csv with information for only 1 product
  else:
    product = product_list[cat_idx]
    df = pd.read_csv('./data/'+product + '.csv')
    df_meta = df

  stemmer = PorterStemmer()
  ## Run Tfidf and find competitors based on description field
  if doCompetitors and not runFromProductCSV:
    print 'Locating competitors for %s...' % product
    ## You only need to fit once
    description_dict,product_id, competitor_urls = \
      competitorTfIdf(df_meta,product,doFit=True)
    description_target = df_meta['description'][df_meta['asin']==product].iloc[0]
    print description_target
    print
    sim_dict = cosineSimilarity(description_target,product)
    top_indices=np.argsort( sim_dict.values() )[::-1][0:10]
    competitors = [(product_id[i],competitor_urls[i]) for i in top_indices]

    with open('competitors_'+product+'.pkl','wb') as handle:
      pickle.dump(competitors,handle)

    with open('description_dict_'+product+'.pkl','wb') as handle:
      pickle.dump(description_dict,handle)

    sys.exit(0)

  ## Plot distribution of user star-ratings
  fig=plt.figure()
  ax = plt.subplot(111)
  df.overall.plot(kind='hist',bins=10,title='Amazon User Ratings:'+product,\
       color='g',alpha=0.5)
  ax.set_xlabel('Star Rating')
  ax.set_ylabel('Users')
  ax.set_xlim(0,5.5)
  plt.grid(True)
  plt.savefig('StarRating_'+product+'.png')
  plt.close()

  #df_subset = df[['reviewText','summary']].reset_index(drop=True)
  reviews = df.reviewText[:int(1.0*len(df.reviewText))]
  summaries = df.summary[:int(1.0*len(df.summary))]
  amazon_stars = df.overall[:int(1.0*len(df.overall))]

  #reviews = df.reviewText[:int(1.0*len(df.reviewText))].sample(n=int(0.95*len(df.reviewText)))
  #summaries = df.summary[:int(1.0*len(df.summary))].sample(n=int(0.95*len(df.summary)))
  #amazon_stars = df.overall[:int(1.0*len(df.overall))].sample(n=int(0.95*len(df.overall)))
  #descriptions = df.overall[:int(1.0*len(df.description))]

  token_dict = {}
  #description_dict = {}
  good_reviews = {}
  bad_reviews = {}
  sentiment_review_list = []
  sentiment_summary_list = []
  #for index,row in df_subset.iterrows():
  i = 0
  for review,summary,stars in zip(reviews,summaries,amazon_stars):
    ## all lowercase and remove punctuation
    review = str(review)
    summary = str(summary)
    #description = str(description)

    ## VADER sentiment analyzer
    vs_review = vaderSentiment(review)
    vs_summary = vaderSentiment(summary)
    sentiment_review_list.append(vs_review)
    sentiment_summary_list.append(vs_summary)

    ## Casing provides valuable information
    ## for sentiment
    lowers = review.lower()
    no_punctuation = lowers.translate(None,string.punctuation)
    #description = description.lower()
    #description = description.translate(None,string.punctuation)

    token_dict[i] = no_punctuation
    #descriptions[i] = description
    review_sentiment = vs_review['compound']
    if review_sentiment > 0.0 and stars > 3.:
      good_reviews[i] = no_punctuation

    elif review_sentiment < 0.0 and stars<= 3.:
      bad_reviews[i] = no_punctuation

    i+=1

  print
  print 'Reading %d user reviews.' % len(token_dict.values())
  print '%d positive reviews.' % len(good_reviews.values())
  print '%d negative reviews.' % len(bad_reviews.values())
  doc_length_good_reviews = [len(rev.split(' ')) for rev in good_reviews.values()]
  print
  if not runFromProductCSV:
    ## Create new column for sentiment using VADER
    df_review = pd.DataFrame(sentiment_review_list)
    df_summary = pd.DataFrame(sentiment_summary_list)
    df_summary = df_summary.rename(columns = {'compound':'compound_summary', \
                                          'neg':'neg_summary',\
                                          'neu':'neu_summary',\
                                          'pos':'pos_summary'})
    df = df.join(df_review,how='left')
    df = df.join(df_summary,how='left')
    df = df.sort_values(by='compound',ascending=False)
    df.to_csv('./data/'+max_product+'.csv')

  df['reviewTime']=pd.to_datetime(df['reviewTime'], format='%m %d, %Y', errors='coerce')
  print df.head()

  ## Plot sentiment over time
  fig=plt.figure()
  ax = fig.add_subplot(111)
  df_time = df[['compound','reviewTime']].sort_values(by='reviewTime')
  df_time.index = df_time['reviewTime']; del df_time['reviewTime'];
  ## Taking daily average
  df_time.resample('D',how='mean',fill_method='ffill').plot()
  ax = plt.gca()
  ax.legend_ = None
  plt.ylabel('Customer Sentiment')
  plt.xlabel('')
  plt.grid(True)
  plt.savefig('sentiment_timeseries_'+product+'.png')
  plt.close()

  ## plot autocorrelation
  fig = plt.figure(figsize=(12,8))
  dta = df_time.resample('D',how='mean',fill_method='ffill')
  ax = fig.add_subplot(111)
  ax = plt.gca()
  ax.legend_ = None
  #dta = df_time.resample('D',how='mean',fill_method='ffill')
  begin,end = getPredictionWindow(cat_idx)
  #fit=sm.tsa.ARIMA(dta,(2,1,2)).fit().plot_predict(begin,end,ax=ax)
  #fit=sm.tsa.ARIMA(dta,(1,1,2)).fit().plot_predict(begin,end,ax=ax)
  fit=sm.tsa.ARIMA(dta,(1,1,0)).fit().plot_predict(begin,end,ax=ax)
  plt.ylabel('Customer Sentiment')
  plt.xlabel('')
  plt.grid(True)
  plt.savefig('sentiment_timeseries_ARIMA_'+product+'.png')
  plt.close()
  sys.exit(0)

  fig = plt.figure(figsize=(12,8))
  ax1 = fig.add_subplot(211)
  fig = sm.graphics.tsa.plot_acf(dta.values.squeeze(), lags=40, ax=ax1)
  ax2 = fig.add_subplot(212)
  ax2.set_ylabel('Customer Sentiment')
  ## fit
  fig = sm.graphics.tsa.plot_pacf(dta, lags=40, ax=ax2)
  #plt.show(block=False)
  ## rolling mean
  fig = plt.figure(figsize=(12,8))
  #ax_time = df_time.resample('D',how='mean',fill_method='ffill').plot(x_compat=True, style='--')
  ax_time =dta.plot(x_compat=True, legend=0)
  #ax_time = df_time.resample('D',how='mean',).plot(x_compat=True, style='--')
  ## Get time difference between first and last observation
  #dt = dta.index[-1]-dta.index[0]
  #dt = (dt/np.timedelta64(1, 'D')).astype(int)
  dt = getPeriods(cat_idx)
  rolling = pd.rolling_mean(dta, dt)
  rolling.plot(ax=ax_time,legend=0,style='-g',lw=2)
  plt.ylabel('Customer Sentiment')
  plt.xlabel('')
  plt.grid(True)
  plt.savefig('sentiment_rolling_mean_'+product+'.png')
  plt.close()
  del fig

  ## perform time series fit of sentiment
  #grid = (slice(0,6,1),slice(0,3,1),slice(0,3,1))
  from scipy.optimize import brute
  ## use only for optimizing paramters (only do once)
  #brute(objfunc, grid, finish=None)
  #fit=sm.tsa.ARIMA(dta,order).fit()
  #plt.show(block=False)


  ## make some plots
  x1 = df['compound'][df.overall<=3.0].tolist()
  x2 = df['compound'][df.overall>3.0].tolist()

  bins = np.linspace(-1.2, 1.2, 20)
  fig = plt.figure()
  plt.hist(x1, bins, alpha=0.5, label='<=3 stars')
  plt.hist(x2, bins, alpha=0.5, label='>3 stars')
  plt.ylim(0.0,5.5e3)
  plt.xlim(-1.2,1.2)
  plt.xlabel('sentiment score')
  plt.legend(loc='upper right')
  plt.savefig('user_sentiment_ratings_'+product+'.png')
  plt.close()
  #plt.show(block=False)
  ###############################
  ## plot ratings
  #fig = plt.figure()
  #ax = df.overall.plot(kind='hist',bins=10,title='Amazon User Ratings')
  #ax.set_xlabel('Rating')
  #ax.set_ylabel('Users')
  #ax.set_xlim(0,5.5)
  #plt.savefig('user_ratings.png')


  print
  print 30*'-'
  print 'PRODUCT :', str(df['asin'][0])
  print 'TITLE :', str(df['title'][0])
  print 'CATEGORY :', df['categories'][0]
  print 'DESCRIPTION :', str(df['description'][0])
  print
  print 30*'-'
  print 'Positive things users are saying.'
  print 30*'-'
  #good_reviews = df.sort_values(by='compound',ascending=False)['reviewText'].values.tolist()
  df_good =  df[df['compound']>0.0]
  good= df_good.sort_values(by='compound',ascending=False)['reviewText'].values.tolist()
  for review in good[:3]:
    print review
    print 30*'-'
  print

  writeCorpusToTxt = False
  if writeCorpusToTxt:
    print 'Writing corpuses to .txt...'
    corpusToTxt(good_reviews,'good_reviews/good_review')
    corpusToTxt(bad_reviews,'bad_reviews/bad_review')
    print 'Done.'
    sys.exit(0)

  print 30*'-'
  print 'Negative things users are saying.'
  print 30*'-'
  #bad_reviews = df.sort_values(by='compound',ascending=True)['reviewText'].values.tolist()
  df_bad =  df[df['compound']<0.0]
  bad = df_bad.sort_values(by='compound',ascending=True)['reviewText'].values.tolist()
  for review in bad[:3]:
    print review
    print 30*'-'
  print
  print
  ## Calcualte tfidf score from training set
  print 'Calculating tf-idf scores. This may take some time...'
  if doTfidf:
    #tfidf = TfidfVectorizer(tokenizer=tokenize,stop_words='english',ngram_range=(2, 3))
    tfidf = TfidfVectorizer(tokenizer=tokenize_nostem,stop_words='english',        ngram_range=(2, 3),lowercase=True,strip_accents="unicode")
    tfs = tfidf.fit_transform(token_dict.values())

    tfidf_good_reviews = TfidfVectorizer(tokenizer=tokenize_nostem,stop_words='english',ngram_range=(2, 3),lowercase=True,strip_accents="unicode")
    #tfidf_good_reviews = TfidfVectorizer(stop_words='english', lowercase=True, strip_accents="unicode", use_idf=True, norm="l2", min_df = 5)
    tfs_good_reviews = tfidf_good_reviews.fit_transform(good_reviews.values())

    #tfidf_bad_reviews = TfidfVectorizer(stop_words='english', lowercase=True, strip_accents="unicode", use_idf=True, norm="l2", min_df = 5)
    tfidf_bad_reviews = TfidfVectorizer(tokenizer=tokenize_nostem,stop_words='english',ngram_range=(2, 3),lowercase=True,strip_accents="unicode")
    tfs_bad_reviews = tfidf_bad_reviews.fit_transform(bad_reviews.values())



    SaveTransform(tfs,'tfidf_transform_'+product)
    SaveVectorizer(tfidf,'Vectorizer_'+product)
    SaveVectors(tfidf.idf_,'tfidf_'+product)
    SaveDictionary(tfidf.vocabulary_,'feature_'+product)

    SaveTransform(tfs_good_reviews,'tfidf_transform_good_reviews_'+product)
    SaveVectorizer(tfidf_good_reviews,'Vectorizer_good_reviews_'+product)
    SaveVectors(tfidf_good_reviews.idf_,'tfidf_good_reviews_'+product)
    SaveDictionary(tfidf_good_reviews.vocabulary_,'feature_good_reviews_'+product)

    SaveTransform(tfs_bad_reviews,'tfidf_transform_bad_reviews_'+product)
    SaveVectorizer(tfidf_bad_reviews,'Vectorizer_bad_reviews_'+product)
    SaveVectors(tfidf_bad_reviews.idf_,'tfidf_bad_reviews_'+product)
    SaveDictionary(tfidf_bad_reviews.vocabulary_,'feature_bad_reviews_'+product)

  else:

    #Import the vectorizer
    tfidf = joblib.load('Vectorizer_'+product+'.pkl')
    tfidf_good_reviews = joblib.load('Vectorizer_good_reviews_'+product+'.pkl')
    tfidf_bad_reviews = joblib.load('Vectorizer_bad_reviews_'+product+'.pkl')

    ## Import transforms
    tfs = joblib.load('tfidf_transform_'+product+'.pkl')
    tfs_good_reviews = joblib.load('tfidf_transform_good_reviews_'+product+'.pkl')
    tfs_bad_reviews = joblib.load('tfidf_transform_bad_reviews_'+product+'.pkl')

    #Import the vocabulary
    #vocab = pickle.load(open("feature.pkl","rb"))

    #Set the vectorizer vocabularly to the imported one above
    #tfidf.vocabulary_ = vocab

    #Import the Tf-Idf matrix
    #tfidf_matrix = joblib.load('tfidf.pkl')

    #Set the trained Tf-Idf matrix
    #trainVectorizerArray = tfidf_matrix.toarray()

  print 'Done!'

  ## Store list of terms for later use
  num_reviews = len(tfidf.vocabulary_)
  num_good_reviews = len(tfidf_good_reviews.vocabulary_)
  num_bad_reviews = len(tfidf_bad_reviews.vocabulary_)

  terms = [""]*num_reviews
  for term in tfidf.vocabulary_.keys():
    terms[tfidf.vocabulary_[term] ] = term

  terms_good_reviews = [""]*num_good_reviews
  for term in tfidf_good_reviews.vocabulary_.keys():
    terms_good_reviews[tfidf_good_reviews.vocabulary_[term] ] = term

  terms_bad_reviews = [""]*num_bad_reviews
  for term in tfidf_bad_reviews.vocabulary_.keys():
    terms_bad_reviews[tfidf_bad_reviews.vocabulary_[term] ] = term

  print "Created document-term matrix for positive reviews of size %d x %d" \
      % (tfs.shape[0],tfs.shape[1])

  print "Created document-term matrix for positive reviews of size %d x %d" \
      % (tfs_good_reviews.shape[0],tfs_good_reviews.shape[1])

  print "Created document-term matrix for positive reviews of size %d x %d" \
      % (tfs_bad_reviews.shape[0],tfs_bad_reviews.shape[1])

  ## quick test
  #test_input = 'This is a very nice product. The pedometer tracks all my steps and keeps me in shape.'
  #print 'Running test with input: %s:' % test_input
  #print 'word stem', ' - ', 'tfidf score'
  #feature_names = tfidf.get_feature_names()
  #response = tfidf.transform([test_input])
  #for col in response.nonzero()[1]:
  #  print feature_names[col], ' - ', response[0,col]

  ## Gensim specific conversions
  dictionary = tfidf.vocabulary_

  n_features = 1000
  #tf_vectorizer = CountVectorizer(tokenizer=tokenize,max_df=0.95,min_df=2,\
                                  #max_features=n_features, stop_words='english', \
                                  #ngram_range=(1,3))

  tf_vectorizer = CountVectorizer(tokenizer=tokenize_nostem,max_df=0.99,min_df=5,\
                                  max_features=n_features, stop_words='english', \
                                  ngram_range=(2,3))

  tf_vectorizer_good_reviews = CountVectorizer(tokenizer=tokenize_nostem,max_df=0.99,min_df=5,\
                                max_features=n_features, stop_words='english', \
                                ngram_range=(2,3))

  tf_vectorizer_bad_reviews = CountVectorizer(tokenizer=tokenize_nostem,max_df=0.99,min_df=5,\
                                  max_features=n_features, stop_words='english', \
                                  ngram_range=(2,3))

  #tf=tf_vectorizer.fit_transform(token_dict.values())
  ## Document term matrices
  tf=tf_vectorizer.fit_transform(good_reviews.values())
  tf_good_reviews=tf_vectorizer_good_reviews.fit_transform(good_reviews.values())
  tf_bad_reviews=tf_vectorizer_bad_reviews.fit_transform(bad_reviews.values())

  doFitLDA = False
  doRunLDA = False
  doFitNMF = False
  doRunNMF = True

  represent_good_reviews_dict = {}
  represent_bad_reviews_dict = {}
  represent_good_reviews = []
  represent_bad_reviews = []
  ## n_topics based on that which maximizes Likelihood
  if doFitLDA and doRunLDA:
    #lda = LatentDirichletAllocation(n_topics=2, max_iter=100, \
                    #learning_method='online', random_state=0, n_jobs=-1)

    lda_good_reviews = LatentDirichletAllocation(n_topics=1, max_iter=100, \
                       learning_method='online', random_state=0, n_jobs=-1)

    lda_bad_reviews = LatentDirichletAllocation(n_topics=1, max_iter=100, \
                       learning_method='online', random_state=0, n_jobs=-1)

    print 'Performing LDA fit...'
    #lda.fit(tf)
    lda_good_reviews.fit(tf_good_reviews)
    lda_bad_reviews.fit(tf_bad_reviews)
    #print 'nll:', lda.score(tf)
    with open('ldamodel_goodreviews_'+product+'.pkl','wb') as handle:
      pickle.dump(lda_good_reviews,handle)
    with open('ldamodel_badreviews_'+product+'.pkl','wb') as handle:
      pickle.dump(lda_bad_reviews,handle)

  elif doRunLDA:
    #lda = joblib.load('ldamodel_'+product+'.pkl')
    lda_good_reviews = joblib.load('ldamodel_goodreviews_'+product+'.pkl')
    lda_bad_reviews = joblib.load('ldamodel_badreviews_'+product+'.pkl')

  elif doFitNMF and doRunNMF:
    print 'Performing NMF fit...'
    n_topics = 4

    nmf_model = decomposition.NMF(init="nndsvd",n_components=n_topics, max_iter=800)
    nmf=nmf_model.fit_transform(tfs)

    H = nmf_model.components_

    nmf_good_reviews_model = decomposition.NMF(init="nndsvd",n_components=n_topics, max_iter=800)
    ## Document topic matrices
    nmf_good_reviews=nmf_good_reviews_model.fit_transform(tfs_good_reviews)

    ## Get indices corresponing to documents with highest
    ## probability for each topic

    #index_good_reviews = nmf_good_reviews.argmax(axis=0)
    for topic_index in range(nmf_good_reviews.shape[1]):
      represent_good_reviews=[]
      ## get top 3 reviews in this topic
      index_good_reviews = np.argsort( nmf_good_reviews[:,topic_index] )[::-1][0:3]
      for ix in index_good_reviews:
        represent_good_reviews.append(good_reviews.values()[ix])
        represent_good_reviews_dict[topic_index] = represent_good_reviews
    #NMF(n_components=n_topics, max_iter=200,\
    #    random_state=1, alpha=.1, l1_ratio=.5).fit(tfidf_good_reviews)

    H_good_reviews = nmf_good_reviews_model.components_

    nmf_bad_reviews_model = decomposition.NMF(init="nndsvd",n_components=n_topics, max_iter=500)
    nmf_bad_reviews = nmf_bad_reviews_model.fit_transform(tfs_bad_reviews)

    #index_bad_reviews = nmf_bad_reviews.argmax(axis=0)
    for topic_index in range(nmf_bad_reviews.shape[1]):
      represent_bad_reviews=[]
      ## get top 3 reviews in this topic
      index_bad_reviews = np.argsort( nmf_bad_reviews[:,topic_index] )[::-1][0:3]
      for ix in index_bad_reviews:
        represent_bad_reviews.append(bad_reviews.values()[ix])
        represent_bad_reviews_dict[topic_index] = represent_bad_reviews
    #NMF(n_components=n_topics, max_iter=200,\
    #    random_state=1, alpha=.1, l1_ratio=.5).fit(tfidf_bad_reviews)

    H_bad_reviews = nmf_bad_reviews_model.components_

    with open('nmf_model_'+product+'.pkl','wb') as handle:
      pickle.dump(nmf_model,handle)

    with open('nmf_goodreviews_model_'+product+'.pkl','wb') as handle:
      pickle.dump(nmf_good_reviews_model,handle)

    with open('nmf_badreviews_model_'+product+'.pkl','wb') as handle:
      pickle.dump(nmf_bad_reviews_model,handle)

    with open('nmf_tf_'+product+'.pkl','wb') as handle:
      pickle.dump(nmf,handle)

    with open('nmf__tf_goodreviews_'+product+'.pkl','wb') as handle:
      pickle.dump(nmf_good_reviews,handle)

    with open('nmf_tf_badreviews_'+product+'.pkl','wb') as handle:
      pickle.dump(nmf_bad_reviews,handle)

    with open('represent_good_reviews_dict_'+product+'.pkl','wb') as handle:
      pickle.dump(represent_good_reviews,handle)

    with open('represent_bad_reviews_dict_'+product+'.pkl','wb') as handle:
      pickle.dump(represent_bad_reviews,handle)


  elif doRunNMF:
    nmf_model = joblib.load('nmf_model_'+product+'.pkl')
    nmf = joblib.load('nmf_tf_'+product+'.pkl')
    H = nmf_model.components_

    nmf_good_reviews_model = joblib.load('nmf_goodreviews_model_'+product+'.pkl')
    nmf_good_reviews = joblib.load('nmf__tf_goodreviews_'+product+'.pkl')
    H_good_reviews = nmf_good_reviews_model.components_

    nmf_bad_reviews_model = joblib.load('nmf_badreviews_model_'+product+'.pkl')
    nmf_bad_reviews = joblib.load('nmf_tf_badreviews_'+product+'.pkl')
    H_bad_reviews = nmf_bad_reviews_model.components_

  #tf_feature_names = tf_vectorizer.get_feature_names()
  tf_feature_names = tf_vectorizer.get_feature_names()
  tf_feature_names_good_reviews = tf_vectorizer_good_reviews.get_feature_names()
  tf_feature_names_bad_reviews = tf_vectorizer_bad_reviews.get_feature_names()

  from wordcloud import WordCloud

  n_top_words = 5
  if doRunNMF:

    print 30*'-'
    print 'Topics in all reviews'
    #print_top_words(nmf, tf_feature_names, n_top_words)
    for topic_index in range( H.shape[0] ):
      top_indices = np.argsort( H[topic_index,:] )[::-1][0:n_top_words]
      term_ranking = [terms[i] for i in top_indices]
      print "Topic %d: %s" % ( topic_index, ", ".join( term_ranking ) )
    print 30*'-'
    print

    print 30*'-'
    print 'Topics in good reviews'
    #print_top_words(nmf_good_reviews, tf_feature_names_good_reviews, n_top_words)
    ## Iterate over each topic
    for topic_index in range( H_good_reviews.shape[0] ):
      ## Grab the indices with largest term weights
      top_indices = np.argsort( H_good_reviews[topic_index,:] )[::-1][0:n_top_words]
      ## Grab the actual weights (sorted in descending order)
      top_weights = sorted(H_good_reviews[topic_index,:])[::-1][0:n_top_words]
      ## Grab the terms corresponding to those weights
      term_ranking = [terms_good_reviews[i] for i in top_indices]
      words = ' '.join(term_ranking)
      wordcloud = WordCloud(background_color='black',width=1800,height=1400).generate(words)
      plt.figure()
      plt.imshow(wordcloud)
      plt.axis('off')
      plt.savefig('./good_reviews_wordcloud_'+product+'_0'+str(topic_index)+'.png', dpi=300)
      plt.close()
      ## One plot per topic
      plt.figure(figsize=(12,10))
      plt.barh(np.arange(len(term_ranking)),top_weights,align='center',color='g',alpha=0.5)
      plt.yticks(np.arange(len(term_ranking)),term_ranking)
      plt.xlabel('weights')
      plt.grid(True)
      plt.tight_layout()
      #plt.show(block=False)
      plt.savefig('positive_reviews_'+product+'_0'+str(topic_index)+'.png')
      plt.close()
      print "Topic %d: %s" % ( topic_index, ", ".join( term_ranking ) )
    print 30*'-'
    print

    print 30*'-'
    print 'Topics in bad reviews'
    for topic_index in range( H_bad_reviews.shape[0] ):
      top_indices = np.argsort( H_bad_reviews[topic_index,:] )[::-1][0:n_top_words]
      term_ranking = [terms_bad_reviews[i] for i in top_indices]

      words = ' '.join(term_ranking)
      wordcloud = WordCloud(background_color='black',width=1800,height=1400).generate(words)
      plt.figure()
      plt.imshow(wordcloud)
      plt.axis('off')
      plt.savefig('./bad_reviews_wordcloud_'+product+'_0'+str(topic_index)+'.png', dpi=300)
      plt.close()
      print "Topic %d: %s" % ( topic_index, ", ".join( term_ranking ) )

      fig=plt.figure(figsize=(12,10))
      plt.barh(np.arange(len(term_ranking)),top_weights,align='center',color='g',alpha=0.5)
      plt.yticks(np.arange(len(term_ranking)),term_ranking)
      plt.xlabel('weights')
      plt.grid(True)
      plt.tight_layout()
      #plt.show(block=False)
      plt.savefig('negative_reviews_'+product+'_0'+str(topic_index)+'.png')
      plt.close()
    #print_top_words(nmf_bad_reviews, tf_feature_names_bad_reviews, n_top_words)
    print 30*'-'

  elif doRunLDA:
    print 30*'-'
    print 'Topics in good reviews'
    print_top_words(lda_good_reviews, tf_feature_names_good_reviews, n_top_words)
    print 30*'-'
    print
    print 30*'-'
    print 'Topics in bad reviews'
    print_top_words(lda_bad_reviews, tf_feature_names_bad_reviews, n_top_words)
    print 30*'-'
    print

  #plt.show(block=False)
  with open('df_'+product+'.pkl','wb') as handle:
    pickle.dump(df,handle)

  with open('df_good_'+product+'.pkl','wb') as handle:
    pickle.dump(df_good,handle)

  with open('df_bad_'+product+'.pkl','wb') as handle:
    pickle.dump(df_bad,handle)

