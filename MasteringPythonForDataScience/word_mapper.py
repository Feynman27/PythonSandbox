#!/usr/bin/python

import sys
for l in sys.stdin:
# remove trailing and leading white space
    l=l.strip()

# split words in the line
    word_tokens=l.split()

# Key-Value pair
    for w in word_tokens:
        print '%s\t%s'% (w,1)
