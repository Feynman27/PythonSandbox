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

# Trending topic on twitter.
q = '#Serena'

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

for label,data in (('Word',words),('Screen Name',screen_names),('Hashtag',hashtags)):
    pt = PrettyTable(field_names=[label,'Count'])
    c = Counter(data)
    [ pt.add_row(kv) for kv in c.most_common()[:10] ]
    pt.align[label], pt.align['Count']='l', 'r' # set column alignment
    print pt


print '============================================================================='
print 'Lexical diversity in words: ', lexical_diversity(words)
print 'Lexical diversity in screen names: ', lexical_diversity(screen_names)
print 'Lexical diversity in hashtags: ', lexical_diversity(hashtags)
print 'Average words per tweet ', average_words(status_texts)
print '============================================================================='
