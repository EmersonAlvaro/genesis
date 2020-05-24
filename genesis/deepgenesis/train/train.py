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

        checkpoint_filepath = str(model_dir.joinpath('model.h5'))
        
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='acc',
            mode='max',
            verbose=1,
            save_best_only=True)

        
        model = xception.get_model()
        adam = tf.keras.optimizers.Adam(lr=lr)

        model.compile(optimizer=adam, loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])    

        history = model.fit(X, Y, epochs=epochs, 
        callbacks=[model_checkpoint_callback],
        verbose=2)

        model_json = model.to_json()
        with open(str(model_dir.joinpath('model.json')), "w") as json_file:
            json_file.write(model_json)
        
        model.save_weights(str(model_dir.joinpath('model.h5')))
        print("Saved model to disk")

        plt.plot(history.history['accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train'], loc='upper left')
        plt.savefig(str(model_dir.joinpath('acc.png')))
        plt.close()


        plt.plot(history.history['loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train'], loc='upper left')
        plt.savefig(str(model_dir.joinpath('loss.png')))
