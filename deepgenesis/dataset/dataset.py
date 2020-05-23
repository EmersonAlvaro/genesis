from settings import *

import tensorflow as tf
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder 
import numpy as np

class DataSet:

    def __init__(self):
        X, Y = self.load_dataset()

        # print(" {} \n input shape {}".format(X, X.shape))
        # print("{} \n labels in one-hot encoding shape {}".format(Y, Y.shape))
       
    def load_dataset(self):
        path = Path.joinpath(data_dir,'symptoms_cancer_test.csv')
        df = pd.read_csv(path)
        
        print(df)
        le = LabelEncoder()
        df.cancertype = le.fit_transform(df.cancertype.values)                   
        
        # ct = ColumnTransformer([("disease", OneHotEncoder(), [1])],    remainder = 'passthrough')
        # df = ct.fit_transform(df)
        X = []
        S = df.symptoms.values
        R = df.redflags.values
        Y = df.cancertype.values

        bitmap_size =  len(S[0].split(' '))
        num_examples = len(S)

        for s, r in zip(S, R):
            s = s.split(' ')
            r = r.split(' ')

            for ch in s+r: 
                X.append(float(ch))
        
        
        X = np.array(X)
        print(X)
        X = X.reshape(num_examples, bitmap_size, 2)
        print(X[3])
        print(X.shape)
        Y = np.array(Y)
        return X, Y