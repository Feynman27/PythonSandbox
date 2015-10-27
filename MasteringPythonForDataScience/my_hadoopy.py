#!/usr/bin/python

import hadoopy
import os

def read_local_dir(local_path):
    for fn in os.listdir(local_path):
        path=os.path.join(local_path,fn)
        if os.path.isfile(path):
            yield path

def main():
    local_path='../data/'
    #hadoopy.writetb('/tmp/data/',read_local_dir(local_path))
    for file in read_local_dir(local_path):
        hadoopy.put(file,'/tmp/data')
        print "The file %s has been put into hdfs" % (file)
if __name__ == '__main__':
    main()
