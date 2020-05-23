from settings import *

import tensorflow as tf
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder 
from sklearn.compose import ColumnTransformer
import numpy as np

class DataSet:

    def __init__(self):
        x, y = self.load_dataset()

        print("====== features ======= \n{} \n".format(x))
        print("======== labels in one-hot encoding ====== \n{}\n".format(y))
       
    def load_dataset(self):
        path = Path.joinpath(data_dir,'onehotenc_symptoms_desease.csv')
        df = pd.read_csv(path)
        
        print("====== entire dataset ===== \n {} ".format(df))

        le = LabelEncoder()
        df.desease = le.fit_transform(df.desease.values)                
        df.symptoms = [np.array(symptoms.split(' '), dtype=np.float32) for symptoms in df.symptoms]

        ct = ColumnTransformer([("desease", OneHotEncoder(), [1])],    remainder = 'passthrough')
        df = ct.fit_transform(df)

        return df[:, -1], df[:, :-1]