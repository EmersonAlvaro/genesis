from settings import *

import tensorflow as tf
import pandas as pd
from pathlib import Path
import numpy as np

class DataSet:

    def __init__(self):
        pass
     
    def load_dataset(self):
        path = Path.joinpath(data_dir,'symptoms_cancer_test.csv')
        df = pd.read_csv(path)
        
        X = []
        S = df.symptoms.values
        R = df.redflags.values
        Y = df.cancertypeid.values

        bitmap_size =  len(S[0].split(' '))
        num_examples = len(S)

        for s, r in zip(S, R):
            s = s.split(' ')
            r = r.split(' ')

            for ch in s+r: 
                X.append(float(ch))
        
        X = np.array(X)
        X = X.reshape(num_examples, 2, bitmap_size)
        X = X.transpose([0, 2, 1])
        Y = np.array(Y)
        
        return X, Y