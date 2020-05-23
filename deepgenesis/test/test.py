from dataset.dataset import DataSet
from  settings import model_dir
from model.xception import Xception

import tensorflow as tf
from pathlib import Path
import numpy as np

class Test:

    def __init__(self):
        self.ds = DataSet()

    def test(self):
        X, Y = self.ds.load_dataset()
        
        print("dataset size {}".format(X.shape))

        json_file = open(str(model_dir.joinpath('model.json')), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        
        model = tf.keras.models.model_from_json(loaded_model_json)
        model.load_weights(str(model_dir.joinpath('model.h5')))
        print("Loaded model from disk")

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        results = model.evaluate(X, Y)
        print('test loss, test acc:', results)

        for i, x in enumerate(X):
            pred = model.predict(np.array([x]))
            print("{}: {}: max: {}".format (i, pred[0][i], np.max(pred[0])))