import twitter
import json

# functions for computing lexical diversity
def lexical_diversity(tokens):
    return 1.0*len(set(tokens))/len(tokens)

# A function for computing avg number of words per tweet
def average_words(statuses):
    total_words = sum([ len(s.split()) for s in statuses ])
    return 1.0*total_words/len(statuses)

CONSUMER_KEY = 'rL1nCCUuzWRym9BxwgVzzcG83'
CONSUMER_SECRET = 'TJjgSFk6DI4eFr5FK7i1kiFvgmwf4yXQ6Ry86Yw95Eg6SMjdk7'
OAUTH_TOKEN = '3232034929-DTeKUUX92QlpBKWMuo6ExEIjFPim8pezLHRCnUX'
OAUTH_TOKEN_SECRET = 'zHOYKpu8ZHm6wpiXuMgK381EjBavuWhsEQyjVoHrFA3eu'

auth=twitter.oauth.OAuth(OAUTH_TOKEN,OAUTH_TOKEN_SECRET,CONSUMER_KEY,CONSUMER_SECRET)
twitter_api = twitter.Twitter(auth=auth)

# The Yahoo! Where On Earth ID for the entire world is 1
# See https://dev.twitter.com/docs/api/1.1/get/trends/place and
# http://developer.yahoo.com/geo/geoplanet/

WORLD_WOE_ID = 1
US_WOE_ID = 23424977

# Prefix ID with the underscore for query string parametrization
# Without the underscore, the twitter package appends the ID value
# to the URL itself as a special case keyword argument

world_trends = twitter_api.trends.place(_id=WORLD_WOE_ID)
us_trends = twitter_api.trends.place(_id=US_WOE_ID)

#print world_trends
#print
#print us_trends
print '============================================================================='
print json.dumps(world_trends,indent=1)
print '============================================================================='
print
print '============================================================================='
print json.dumps(us_trends,indent=1)
print '============================================================================='

world_trends_set = set([trend['name'] for trend in world_trends[0]['trends'] ])
us_trends_set = set([trend['name'] for trend in us_trends[0]['trends'] ])
common_trends = world_trends_set.intersection(us_trends_set)
print '============================================================================='
print common_trends
print '============================================================================='

# Trending topic on twitter.
q = '#Happy 4th of July'

count = 100

# See https://dev.twitter.com/docs/api/1.1/get/search/tweets
search_results = twitter_api.search.tweets(q=q,count=count)
statuses = search_results['statuses']

# Iterate through 5 more batches of results by following the cursor
for _ in range(5):
    print "Length of statuses ", len(statuses)
    try:
        next_results = search_results['search_metadata']['next_results']
    except KeyError, e: # No more results when next_results does not exist
        break

# Create a dictionary from next_results
kwargs = dict([ kv.split('=') for kv in next_results[1:].split("&") ])

search_results = twitter_api.search.tweets(**kwargs)
statuses += search_results['statuses']

# Show one sample search result by slicing the list
print '============================================================================='
print json.dumps(statuses[0],indent=1)

# extract text from tweets
status_texts = [ status['text']
                    for status in statuses ]
# extract screen names
screen_names = [ user_mention['screen_name']
                    for status in statuses
                        for user_mention in status['entities']['user_mentions'] ]

hashtags = [ hashtag['text']
                for status in statuses
                    for hashtag in status['entities']['hashtags']]

# Compute collection of all words from all tweets
words = [ w
            for t in status_texts
                for w in t.split() ]

# Explore first 5 items for each
print '============================================================================='
print json.dumps(status_texts[:5],indent=1)
print json.dumps(screen_names[:5],indent=1)
print json.dumps(hashtags[:5],indent=1)
print json.dumps(words[:5],indent=1)
print '============================================================================='

from collections import Counter
for item in [words,screen_names,hashtags]:
    c=Counter(item)
    print c.most_common()[:10] # top 10
    print
print '============================================================================='

from prettytable import PrettyTable
import matplotlib.pyplot as plt

for label,data in (('Word',words),('Screen Name',screen_names),('Hashtag',hashtags)):
    pt = PrettyTable(field_names=[label,'Count'])
    c = Counter(data)
    [ pt.add_row(kv) for kv in c.most_common()[:10] ]
    pt.align[label], pt.align['Count']='l', 'r' # set column alignment
    print pt
    plt.hist(c.values())
    plt.title(label)
    plt.ylabel('Number of items/bin')
    plt.xlabel('frequency')
    plt.show()


print '============================================================================='
print 'Lexical diversity in words: ', lexical_diversity(words)
print 'Lexical diversity in screen names: ', lexical_diversity(screen_names)
print 'Lexical diversity in hashtags: ', lexical_diversity(hashtags)
print 'Average words per tweet ', average_words(status_texts)
print '============================================================================='

retweets = [
                # store tuple of three values
                (status['retweet_count'], status['retweeted_status']['user']['screen_name'], status['text'])
                for status in statuses if status.has_key('retweeted_status') ]

# Slice of first 5 from sorted results and display each item in tuple
pt = PrettyTable(field_names=['Count','Screen Name', 'Text'])
[pt.add_row(row) for row in sorted(retweets,reverse=True)[:5] ]
pt.max_width['Text'] = 50
pt.align = 'l'
print '============================================================================='
print pt
counts = [count for count, _, _ in retweets]
plt.hist(counts)
plt.xlabel('frequency')
plt.ylabel('number of retweets/bin')
plt.show()
print counts
print '============================================================================='

_retweets = twitter_api.statuses.retweets(id=617371618835320832)
print [r['user']['screen_name'] for r in _retweets]
print '============================================================================='

word_counts = sorted(Counter(words).values(),reverse=True)
plt.loglog(word_counts)
plt.ylabel('Freq')
plt.xlabel('Word Rank')
plt.show()
