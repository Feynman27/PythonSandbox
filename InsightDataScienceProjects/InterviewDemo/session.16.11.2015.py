# coding: utf-8
complete_dataset_url = 'http://files.grouplens.org/datasets/movielens/ml-latest.zip'
small_dataset_url = 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
import os
datasets_path = os.path.join('..', 'datasets')
complete_dataset_path = os.path.join(datasets_path, 'ml-latest.zip')
small_dataset_path = os.path.join(datasets_path, 'ml-latest-small.zip')
complete_dataset_path
small_dataset_path
import urllib
small_f = urllib.urlretrieve (small_dataset_url, small_dataset_path)
complete_f = urllib.urlretrieve (complete_dataset_url, complete_dataset_path)
import zipfile
with zipfile.ZipFile(small_dataset_path, "r") as z: z.extractall(datasets_path)
with zipfile.ZipFile(complete_dataset_path, "r") as z: z.extractall(datasets_path)
