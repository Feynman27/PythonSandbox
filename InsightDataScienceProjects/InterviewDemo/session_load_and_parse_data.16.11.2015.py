# coding: utf-8
## Load raw ratings data.
import os
datasets_path = os.path.join('..', 'datasets')
print datasets_path
small_ratings_file = os.path.join(datasets_path, 'ml-latest-small', 'ratings.csv')
small_ratings_file
from pyspark import SparkContext
sc = SparkContext(appName="CollaborativeFilter")
small_ratings_raw_data = sc.textFile(small_ratings_file)
## Filter header
small_ratings_raw_data_header = small_ratings_raw_data.take(1)[0]

## parse raw data into new RDD
small_ratings_data = small_ratings_raw_data.filter(lambda line: line!=small_ratings_raw_data_header).map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1],tokens[2])).cache()
## print first few lines of result
small_ratings_data.take(3)

## repeat with movies.csv
small_movies_file = os.path.join(datasets_path, 'ml-latest-small', 'movies.csv')

small_movies_raw_data = sc.textFile(small_movies_file)
small_movies_raw_data_header = small_movies_raw_data.take(1)[0]

small_movies_data = small_movies_raw_data.filter(lambda line: line!=small_movies_raw_data_header).map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1])).cache()

small_movies_data.take(3)
