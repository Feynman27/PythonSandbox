import facebook
import json

# A helper function to pretty-print Python objects as JSON
def pp(o):
    print json.dumps(o,indent=1)

# Create a connection to the Graph API with your access token

ACCESS_TOKEN = 'CAACEdEose0cBALtboCin4KD7UoO9W42BJTJxqmdIWmtZBrRAdoK3V9tY5G7khtqP7VGdDJ4BWWuzTQGU2CdBPLBjjuxoF0uHnitl2RvkZBrL6QyCeLvRYwNeZBJjyzczVlsjQIZA94vTtoOUkYERW8l2xh7s5LTuIa81vcHHQxmmJSAxxKs5nATimWdaXc5GDm064F3HpSUtPKQmbLkpwAGLjUNugXUZD'

g = facebook.GraphAPI(ACCESS_TOKEN)

# First, let's query for all of the likes in your social
# network and store them in a slightly more convenient
# data structure as a dictionary keyed on each friend's
# name. We'll use a dictionary comprehension to iterate
# over the friends and build up the likes in an intuitive
# way, although the new "field expansion" feature could
# technically do the job in one fell swoop as follows:
#
#pp(g.get_object('me', fields='id,name,friends.fields(id,name,likes)'))
#
friends = g.get_connections("me", "friends")['data']

likes = { friend['name'] : g.get_connections(friend['id'], "likes")['data']
                  for friend in friends }

print likes
