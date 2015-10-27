#!/usr/bin/python

'''
Calculate the sentiment score for segments of a review.
'''

import sys
import re

def sentiment_score(text,pos_list,neg_list):
    pos_score=0
    neg_score=0

    for w in text.split(' '):
        if w in pos_list: pos_score+=1
        if w in neg_list: neg_score+=1
    return pos_score-neg_score

positive_words=open('../data/positive-words.txt').read().split('\n')
negative_words=open('../data/negative-words.txt').read().split('\n')

for l in sys.stdin:
# strip leading and trailing whitespace
    l=l.strip()

    l=l.lower()

    score=sentiment_score(l,positive_words,negative_words)

    print '%s\t%s' % (l,score)
