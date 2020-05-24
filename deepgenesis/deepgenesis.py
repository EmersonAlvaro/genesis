from dataset.dataset import DataSet
from  settings import model_dir, data_dir
from model.xception import Xception

import tensorflow as tf
from pathlib import Path
import numpy as np
import pandas as pd
from collections import namedtuple

class DeepGenesis:

    def __init__(self):
        pass

    def predict(self, symptoms, redflags):
        x = []
        
        for ch in symptoms+redflags: 
                x.append(float(ch))

        x = np.array(x)
        x = x.reshape(2, len(symptoms))
        x = x.transpose(1,0)

        path = Path.joinpath(data_dir,'cancertypes.csv')
        df = pd.read_csv(path)
        Y = df.cancertypename
        
        json_file = open(str(model_dir.joinpath('model.json')), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        
        model = tf.keras.models.model_from_json(loaded_model_json)
        model.load_weights(str(model_dir.joinpath('model.h5')))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        pred = model.predict(np.array([x]))
        print("max probability: {}".format (np.max(pred[0])))

        results = []

        for i, prob in enumerate(pred[0]):
            result = namedtuple('result', ['id', 'name', 'probability'])
            result.id = i
            result.name = Y[i]
            result.probability = prob
            results.append(result)

        results.sort(key=lambda x: x.probability, reverse=True)  

        return results