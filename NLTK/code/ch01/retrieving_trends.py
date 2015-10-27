import twitter
import json

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
