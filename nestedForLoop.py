#!/usr/bin/python

lList = ['1','2','3']

for i in lList:
    for j in lList:
        product = int(i)*int(j)
        if(product>3): break
        print 'i x j = : ' + str(product)

