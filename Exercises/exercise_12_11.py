#!/usr/bin/python

def divideBy100(number):
    return float(number/100)

my_in = float(raw_input ('Enter a number: '))
newnumber = divideBy100(my_in)
print 'Your number divided by 100 is ' + str(newnumber)
