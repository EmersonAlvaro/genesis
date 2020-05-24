import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf
import pandas as pd
from pathlib import Path

from dataset.dataset import DataSet
from train.train import Train
from test.test import Test
from model.xception import Xception
from deepgenesis import DeepGenesis
from settings import *

'''
    This project uses tensorflow 2.1.0
'''

print('tensorflow version {}'.format(tf.__version__))

ds = DataSet()
xception =  Xception()

# xception.plot_model()
# train = Train()
# train.train(lr=1e-3,epochs=100)

# test = Test()
# test.test()

"""
    Example: Prediciting Leukemia

"""

path = Path.joinpath(data_dir,'dataset.csv')
df = pd.read_csv(path)
symptoms = df.symptoms.values 
redflags = df.redflags.values
        

dg = DeepGenesis()

#symptoms[8] and redflags[8] are strings separeted by simple space
 
results = dg.predict(symptoms=symptoms[8].split(' '), redflags=redflags[8].split(' '))

"""
    results is a list of named tuple in decreasing order of probability 
    with attributes : id, name, probability, where:
    id: is the numerical identification of a cancer
    name: the name of the cancer
    probability: it is the probability of being this cancer type given the symptoms and redflags
"""

for result in results:
    print("{}, {}, {}\n".format(result.id,result.name,result.probability))

print('since the model learnt how to predict Leukemia, it assigned high probability to Leukemia!')