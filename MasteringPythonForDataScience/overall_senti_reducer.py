#!/usr/bin/python

from operator import itemgetter
import sys

total_score=0

for l in sys.stdin:
# input from mapper
    key,score = l.split('\t',1)
    try:
        score=int(score)

    except ValueError: continue

    # update total score from individual scores
    total_score+=score

print '%s' % (total_score)
