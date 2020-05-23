'''
    This project uses tensorflow 2.1.0
'''
import tensorflow as tf
from dataset.dataset import DataSet
from train.train import Train

print('tensorflow version {}'.format(tf.__version__))

ds = DataSet()