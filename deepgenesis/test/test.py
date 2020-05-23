from dataset.dataset import DataSet
from  settings import model_dir
from model.xception import Xception

import tensorflow as tf
from pathlib import Path

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

        # score = loaded_model.evaluate(X, Y, verbose=0)
        # print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))