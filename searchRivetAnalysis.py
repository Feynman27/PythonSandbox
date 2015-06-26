#!/usr/bin/python

my_file = "jobOptions.rivet.test.py"
analysisFile = "analyses.txt"
#analysisFile = "test.txt"
infileAnaly = open(analysisFile,"r")
#analysis = 'MC_TTBAR'
#analysisList = set() 

for analysis in infileAnaly:
  sAnalysis = str(analysis.rstrip())
#  print sAnalysis
    #analysisList.add(analysis)
  found = 0
  infile = open(my_file,"r")
  for line in infile:
#    print line.rstrip()
    found += line.count(sAnalysis)
    if(found>0):
      break
#    if (found == 0):
#        print 'Rivet analysis '+str(analysis)+' was not found.',
#  if (int(found) != 0):
#      print 'Rivet analysis ' + sAnalysis +' was found.'
  if (found == 0):
    print 'Rivet analysis ' + sAnalysis +' NOT found.'
