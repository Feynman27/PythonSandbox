#!/usr/bin/python

'''
Calculate the overall sentiment score for a review.
'''
import sys
import hashlib

def sentiment_score(text,pos_list,neg_list):
    pos_score=0
    neg_score=0

    for w in text.split(' '):
        if w in pos_list: pos_score+=1
        if w in neg_list: neg_score+=1
    return pos_score-neg_score


positive_words=open('./positive-words.txt').read().split('\n')
negative_words=open('./negative-words.txt').read().split('\n')

for l in sys.stdin:
    l=l.strip()
    l=l.lower()

    score=sentiment_score(l,positive_words,negative_words)

    hash_object=hashlib.md5(l)

    print '%s\t%s' % (hash_object.hexdigest(), score)
