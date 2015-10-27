import twitter

CONSUMER_KEY = 'rL1nCCUuzWRym9BxwgVzzcG83'
CONSUMER_SECRET = 'TJjgSFk6DI4eFr5FK7i1kiFvgmwf4yXQ6Ry86Yw95Eg6SMjdk7'
OAUTH_TOKEN = '3232034929-DTeKUUX92QlpBKWMuo6ExEIjFPim8pezLHRCnUX'
OAUTH_TOKEN_SECRET = 'zHOYKpu8ZHm6wpiXuMgK381EjBavuWhsEQyjVoHrFA3eu'

auth=twitter.oauth.OAuth(OAUTH_TOKEN,OAUTH_TOKEN_SECRET,CONSUMER_KEY,CONSUMER_SECRET)
twitter_api = twitter.Twitter(auth=auth)

# Nothing to see
print twitter_api
