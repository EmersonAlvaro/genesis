import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf
from dataset.dataset import DataSet
from train.train import Train
from test.test import Test
from model.xception import Xception

'''
    This project uses tensorflow 2.1.0
'''

print('tensorflow version {}'.format(tf.__version__))

ds = DataSet()
xception =  Xception()

# xception.plot_model()
train = Train()
train.train(epochs=3)