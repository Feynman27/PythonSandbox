import facebook
import json

# A helper function to pretty-print Python objects as JSON
def pp(o):
    print json.dumps(o,indent=1)

# Create a connection to the Graph API with your access token

ACCESS_TOKEN = 'CAACEdEose0cBALtboCin4KD7UoO9W42BJTJxqmdIWmtZBrRAdoK3V9tY5G7khtqP7VGdDJ4BWWuzTQGU2CdBPLBjjuxoF0uHnitl2RvkZBrL6QyCeLvRYwNeZBJjyzczVlsjQIZA94vTtoOUkYERW8l2xh7s5LTuIa81vcHHQxmmJSAxxKs5nATimWdaXc5GDm064F3HpSUtPKQmbLkpwAGLjUNugXUZD'
g = facebook.GraphAPI(ACCESS_TOKEN)

# Execute queries
print '---------------'
print 'Me'
print '---------------'
pp(g.get_object('me'))
print '---------------'
print 'My Friends'
print '---------------'
pp(g.get_connections('me','friends'))
print
print '---------------'
print 'Social Web'
print '---------------'
#pp(g.request("search",{'q' : 'social web', 'type':'page'}))
pp(g.request("search",{'q' : 'Programming Collective Intelligence', 'type':'page'}))
