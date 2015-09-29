#!/usr/bin/python

def divideBy100(number):
    return int(number/100)

my_in = int(raw_input ('Enter a number: '))
newnumber = divideBy100(my_in)
print 'Your number divided by 100 is ' + str(newnumber)
