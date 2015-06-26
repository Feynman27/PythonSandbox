#!/usr/bin/python

from random import sample
my_rand_list = sample(range(100),10)
print 'Random list of numbers:', my_rand_list
largest = -1
for item in my_rand_list:
    if(item>largest):
        largest = item
        print largest,
print
print 'largest number is ', largest
