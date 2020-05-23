import tensorflow as tf
from  settings import model_dir
from pathlib import Path

class Xception:

    def __init__(self, num_classes=26, num_features=30):
        self.inputs = tf.keras.layers.Input(shape=(num_features, 1))
        hiddens = self.build_xception(self.inputs)
        hiddens = tf.keras.layers.Flatten() (hiddens)
        self.outputs = tf.keras.layers.Dense(num_classes, activation=tf.nn.sigmoid,
                  kernel_initializer=tf.initializers.GlorotUniform())(hiddens)

    def get_model(self):
        return tf.keras.Model(self.inputs, self.outputs)

    def build_xception(self, inputs):
        inputs = self.build_entry_flow(inputs)  
        inputs = self.build_middle_flow(inputs)
        inputs = self.build_exit_flow(inputs)  
        return inputs

    def build_entry_flow(self, inputs):
        inputs = self.do_conv1d(inputs=inputs, filters=32, strides=2 , kernel_size=3)
        inputs = self.do_conv1d(inputs=inputs, filters=64, kernel_size=3)
        
        short_cut = self.do_conv1d(inputs=inputs, filters=128, kernel_size=1, useReLu=False)
        inputs = self.do_separableConv1D(inputs=inputs, filters=128, kernel_size=3)
        inputs = self.do_separableConv1D(inputs=inputs, filters=128, kernel_size=3, useReLu=False)
        inputs = self.add([inputs, short_cut])
        
        inputs = self.do_separableConv1D(inputs=inputs, filters=256, kernel_size=3)
        inputs = self.do_separableConv1D(inputs=inputs, filters=256, kernel_size=3, useReLu=False)
        short_cut = self.do_conv1d(inputs=short_cut, filters=256, kernel_size=1, useReLu=False) 
        inputs = self.add([inputs, short_cut])

        inputs = self.do_separableConv1D(inputs=inputs, filters=768, kernel_size=3)
        inputs = self.do_separableConv1D(inputs=inputs, filters=768, kernel_size=3, useReLu=False)
        short_cut = self.do_conv1d(inputs=short_cut, filters=768, kernel_size=1, useReLu=False) 
        inputs = self.add([inputs, short_cut])

        return inputs

    def build_middle_flow(self, inputs):
        short_cut = inputs
        
        for __ in range(8):
            inputs = self.do_separableConv1D(inputs=inputs, filters=768, kernel_size=3)
            inputs = self.do_separableConv1D(inputs=inputs, filters=768, kernel_size=3)
            inputs = self.do_separableConv1D(inputs=inputs, filters=768, kernel_size=3, useReLu=False)
            short_cut = self.do_conv1d(inputs=short_cut, filters=768, kernel_size=1, useReLu=False) 
            inputs = self.add([inputs, short_cut])
            
        return inputs

    def build_exit_flow(self, inputs):
        short_cut = inputs

        inputs = self.do_separableConv1D(inputs=inputs, filters=768, kernel_size=3)
        inputs = self.do_separableConv1D(inputs=inputs, filters=768, kernel_size=3, useReLu=False)
        short_cut = self.do_conv1d(inputs=short_cut, filters=768, kernel_size=1, useReLu=False) 
        inputs = self.add([inputs, short_cut], useReLu=False)
        
        inputs = self.do_separableConv1D(inputs=inputs, filters=768, kernel_size=3)
        inputs = self.do_separableConv1D(inputs=inputs, filters=768, kernel_size=3)
        
        return inputs

    def do_conv1d(self, inputs, filters, kernel_size, strides=1, padding='SAME', useReLu=True):
        outputs = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size,
                        strides=strides, padding=padding,
                        kernel_initializer=tf.initializers.GlorotUniform(),
                        use_bias=False)(inputs)
        outputs = tf.keras.layers.BatchNormalization() (outputs)

        if useReLu:
            return tf.keras.layers.ReLU()(outputs)
        else:
            return outputs

    def do_separableConv1D(self, inputs, filters, kernel_size, strides=1, padding='SAME', useReLu=True):
        outputs = tf.keras.layers.SeparableConv1D(filters=filters, kernel_size=kernel_size,
                        strides=strides, padding=padding,
                        use_bias=False)(inputs)
        outputs = tf.keras.layers.BatchNormalization() (outputs)    
        
        if useReLu:
            return tf.keras.layers.ReLU()(outputs)
        else:
            return outputs
    
    def add(self, inputs, useReLu=True):
        outputs = tf.keras.layers.Add() (inputs)
        
        if useReLu:
            return tf.keras.layers.ReLU()(outputs)
        else:
            return outputs
    
    def plot_model(self):
        model = self.get_model()
        model.summary()
        img_file = Path.joinpath(model_dir, 'Xception.png')
        tf.keras.utils.plot_model(model, to_file=img_file, show_shapes=True)