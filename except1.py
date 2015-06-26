#!/usr/bin/python
string = './test_file.txt'
try:
    f = open(string,'r')
except IOError:
    print "Cannot open the requested file: " + string
    exit()
print "Processing file."
    
