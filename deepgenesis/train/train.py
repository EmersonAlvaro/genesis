from dataset.dataset import DataSet
from  settings import model_dir
from model.xception import Xception

import tensorflow as tf
from pathlib import Path

class Train:
    
    def __init__(self):
        self.ds = DataSet()

    def train(self, lr=1e-3, epochs = 12):
        X, Y = self.ds.load_dataset()
        
        batch , num_features , num_channel = X.shape
        xception = Xception(num_classes=4, num_features=num_features)

        mc = tf.keras.callbacks.ModelCheckpoint(str(model_dir.joinpath('model.h5')), monitor='train_acc',
                             mode='auto', verbose=1, save_best_only=True)
        
        model = xception.get_model()
        adam = tf.keras.optimizers.Adam(lr=lr)

        model.compile(optimizer=adam, loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])    
        history = model.fit(X, Y, epochs=epochs, verbose=2)