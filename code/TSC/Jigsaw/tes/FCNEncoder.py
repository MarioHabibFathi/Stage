#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 10:11:51 2023

@author: mariohabibfathi
"""

import tensorflow as tf
import numpy as np

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# class FCN:

#     def __init__(self,input_shape,n_classes,output_directory,alpha=1e-2,epochs=1000,batch_size=32):

#         self.input_shape = (input_shape,)
#         self.n_classes = n_classes
#         self.output_directory = output_directory
#         self.alpha = alpha
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.build_model()
class FCN:
    def __init__(self, input_shape, output_directory, alpha=1e-2, epochs=1000, batch_size=32):
        self.input_shape = (input_shape,)
        # self.n_classes = n_classes
        self.output_directory = output_directory
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.build_model()

    def build_model(self):
        self.input_layer = tf.keras.layers.Input(self.input_shape)
        self.input_layer_exp = tf.expand_dims(self.input_layer, axis=-1)
        # print(self.input_shape)
        # self.input_layer = tf.keras.layers.Reshape(target_shape = (self.input_shape,1), input_shape = self.input_shape)
        # print(self.input_shape)

        # Add encoder layers
        self.conv1 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, padding='same')(self.input_layer_exp)
        self.bn1 = tf.keras.layers.BatchNormalization()(self.conv1)
        self.relu1 = tf.keras.layers.Activation(activation='relu')(self.bn1)

        self.conv2 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same')(self.relu1)
        self.bn2 = tf.keras.layers.BatchNormalization()(self.conv2)
        self.relu2 = tf.keras.layers.Activation(activation='relu')(self.bn2)

        self.conv3 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same')(self.relu2)
        self.bn3 = tf.keras.layers.BatchNormalization()(self.conv3)
        self.relu3 = tf.keras.layers.Activation(activation='relu')(self.bn3)

        # self.gap = tf.keras.layers.GlobalAveragePooling1D()(self.relu3)

        # Add jigsaw puzzle layers
        # ...

        # Add decoder layers
        self.up4 = tf.keras.layers.UpSampling1D(size=2)(self.relu3)
        self.conv_up4 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same')(self.up4)
        self.bn_up4 = tf.keras.layers.BatchNormalization()(self.conv_up4)
        self.relu_up4 = tf.keras.layers.Activation(activation='relu')(self.bn_up4)

        self.up5 = tf.keras.layers.UpSampling1D(size=2)(self.relu_up4)
        self.conv_up5 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same')(self.up5)
        self.bn_up5 = tf.keras.layers.BatchNormalization()(self.conv_up5)
        self.relu_up5 = tf.keras.layers.Activation(activation='relu')(self.bn_up5)

        self.up6 = tf.keras.layers.UpSampling1D(size=2)(self.relu_up5)
        self.conv_up6 = tf.keras.layers.Conv1D(filters=1, kernel_size=3, padding='same')(self.up6)

        # self.output_layer = tf.squeeze(self.conv_up6, axis=-1)#(self.conv_up6)
        # self.output_layer = tf.keras.layers.Activation("linear")(self.output_layer)
        # self.output_layer = tf.keras.layers.Conv1D(self.input_shape[0],kernel_size=1, activation="linear")(self.conv_up6)


        # Global Average Pooling layer to reduce sequence length to 1
        self.gap = tf.keras.layers.GlobalAveragePooling1D()(self.conv_up6)
        
        # Dense layer with 512 units for final output
        self.dense = tf.keras.layers.Dense(units=self.input_shape[0])(self.gap)
        
        self.output_layer = tf.keras.layers.Activation("linear")(self.dense)

        self.model = tf.keras.models.Model(inputs=self.input_layer, outputs=self.output_layer)

        my_learning_rate = 0.001
        my_optimizer = tf.keras.optimizers.Adam(learning_rate=my_learning_rate)

        my_loss = tf.keras.losses.MeanSquaredError()
        # my_loss = tf.keras.losses.CategoricalCrossentropy()


        self.model.compile(loss=my_loss, optimizer=my_optimizer)
        tf.keras.utils.plot_model(self.model, to_file=self.output_directory+"_model.pdf", show_shapes=True)
        
    def fit(self,xtrain,xval,ytrain,yval):
            
        min_loss = 1e9
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=min_loss)

        loss = []
        val_loss = []

        for _epoch in range(self.epochs):

            # ref_train, pos_train, neg_train = triplet_generation(xtrain)
            # ref_val, pos_val, neg_val = triplet_generation(xval)

            # print(xtrain.shape)
            
            # hist = self.model.fit(xtrain,np.zeros(shape=xtrain.shape),
            #                        epochs=1,batch_size=self.batch_size,verbose=False,
            #                        callbacks=[reduce_lr],validation_data=xval,
            #                        validation_batch_size=self.batch_size,)
            # tf.expand_dims(xtrain, axis=0).shape.as_list()
            # tf.expand_dims(xval, axis=0).shape.as_list()

            # print(xtrain[1].shape)
            # print(xval.shape)
            # print(ytrain.shape)
            # print(yval.shape)

            # type(xtrain.get_shape().as_list())
            # type(xval.get_shape().as_list())

            # print(self.n_classes)            
            # print(yval.shape)
            
            hist = self.model.fit(xtrain,ytrain,
                                   epochs=1,batch_size=self.batch_size,verbose=False,
                                   callbacks=[reduce_lr],validation_data=(xval,yval),
                                   validation_batch_size=self.batch_size,)
            
            loss.append(hist.history['loss'][0])
            val_loss.append(hist.history['val_loss'][0])

        self.model.save(self.output_directory+'last_model.hdf5')
            
    def predict(self,x):

        return self.model.predict(x)