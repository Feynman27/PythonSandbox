#!/usr/bin/python
from __future__ import division
from collections import Counter
import math, random, csv, json

if __name__ == "__main__":

    def process(cat):
        print cat
    with open('../Data_consumer_expenditure_survey.csv', 'rU') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            age = int(row["age"])
            process(age)

