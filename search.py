#!/usr/bin/python

keywords=[]
keywords = ['red','green','blue','yellow']
mySet = set()
if 'yellow' in keywords and any(k in keywords for k in ['black','white']):
    mySet.add('success')
else:
     mySet.add('failure')
print 'Found: ' + str(mySet)

