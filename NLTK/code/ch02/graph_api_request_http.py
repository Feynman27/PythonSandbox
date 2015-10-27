import requests
import json

ACCESS_TOKEN = 'CAACEdEose0cBAA7jpfoGkJmGzLSJS8yUKZA1ZBxabiJRmNHsWhusloIhXnsqr7MUQqEpd1ToEtb8dZC1OdrNSkTkQZBVh8scxT2TRdqb1ApGfFm2ZChgxZAOEP6V4qbVYW1jpITdtxfakMZC1EEaFSyKrww6Wm1dMikBHwzNrxLNsYoMEpbtKaUdRUaTHHAgYrtGRU4RQVoPFXonF5XFKPmCIQHdGiX75gZD'

base_url = 'https://graph.facebook.com/me'

# Get 10 likes for 10 friends
fields = 'id,name,friends.limit(10).fields(likes.limit(10))'

url = '%s?fields=%s&access_token=%s' % \
        (base_url,fields,ACCESS_TOKEN)

print url

# Interpret the response as JSON and convert back
# to Python data structures

content = requests.get(url).json()

# Pretty-print the JSON and display it
print json.dumps(content,indent=1)
