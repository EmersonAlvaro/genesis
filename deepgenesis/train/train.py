from dataset.dataset import DataSet
from  settings import model_dir
from model.xception import Xception

import tensorflow as tf
from pathlib import Path
from matplotlib import pyplot as plt

class Train:
    
    def __init__(self):
        self.ds = DataSet()

    def train(self, lr=1e-3, epochs = 12):
        X, Y = self.ds.load_dataset()
        
        print("dataset size {}".format(X.shape))

        batch , num_features , num_channel = X.shape
        xception = Xception(num_classes=13, num_features=num_features)

        mc = tf.keras.callbacks.ModelCheckpoint(str(model_dir.joinpath('model.h5')), monitor='train_acc',
                             mode='max', verbose=1, save_best_only=True)
        
        model = xception.get_model()
        adam = tf.keras.optimizers.Adam(lr=lr)

        model.compile(optimizer=adam, loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])    

        history = model.fit(X, Y, epochs=epochs, callbacks=[mc] ,verbose=2)

        plt.plot(history['acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train'], loc='upper left')
        plt.close()

        plt.plot(history['loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train'], loc='upper left')
        plt.close()