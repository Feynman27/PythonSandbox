#!/usr/bin/python

from operator import itemgetter
import sys

# keep track of current word being counted
current_word_token=None
counter=0
word=None

#stdin
for l in sys.stdin:
    # remove trailing and leading white space
    l=l.strip()

# parse input from mapper
    word_token,counter=l.split('\t',1)
# convert counter to int
    try:
        counter=int(counter)
# ignore line if counter is not type int
    except ValueError: continue

    if current_word_token == word_token:
        current_counter+=counter
    else:
        if current_word_token:
            print '%s\t%s' % (current_word_token,current_counter)

        current_counter = counter
        current_word_token=word_token

if current_word_token == word_token:
    print '%s\t%s' % (current_word_token,current_counter)

