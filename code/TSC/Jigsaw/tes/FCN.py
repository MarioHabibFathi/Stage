#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 10:10:05 2023

@author: mariohabibfathi
"""

import tensorflow as tf
import numpy as np

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class FCN:

    def __init__(self,input_shape,n_classes,output_directory,alpha=1e-2,epochs=1000,batch_size=32):

        self.input_shape = (input_shape,)
        self.n_classes = n_classes
        self.output_directory = output_directory
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.build_model()
    
    def build_model(self):

        self.input_layer = tf.keras.layers.Input(self.input_shape)

        self.reshape = tf.keras.layers.Reshape(target_shape=(self.input_shape[0],1),name='fff')(self.input_layer)

        self.conv1 = tf.keras.layers.Conv1D(filters=128,kernel_size=8,padding='same')(self.reshape)
        self.bn1 = tf.keras.layers.BatchNormalization()(self.conv1)
        self.relu1 = tf.keras.layers.Activation(activation='relu')(self.bn1)

        self.conv2 = tf.keras.layers.Conv1D(filters=256,kernel_size=5,padding='same')(self.relu1)
        self.bn2 = tf.keras.layers.BatchNormalization()(self.conv2)
        self.relu2 = tf.keras.layers.Activation(activation='relu')(self.bn2)

        self.conv3 = tf.keras.layers.Conv1D(filters=128,kernel_size=3,padding='same')(self.relu2)
        self.bn3 = tf.keras.layers.BatchNormalization()(self.conv3)
        self.relu3 = tf.keras.layers.Activation(activation='relu')(self.bn3)

        self.gap = tf.keras.layers.GlobalAveragePooling1D()(self.relu3)

        # self.output_shape = tf.keras.layers.Reshape(target_shape=(self.n_classes,1))(self.gap)
        # self.output_layer = tf.keras.layers.Activation("softmax")(self.gap)

        self.output_layer = tf.keras.layers.Dense(self.n_classes, activation="softmax")(self.gap)

        self.model = tf.keras.models.Model(inputs=self.input_layer,outputs=self.output_layer)
        
        
        my_learning_rate = 0.001
        my_optimizer = tf.keras.optimizers.Adam(learning_rate=my_learning_rate)
        
        # my_loss = triplet_loss_function(alpha=self.alpha)
        
        my_loss = tf.keras.losses.CategoricalCrossentropy()

        
        self.model.compile(loss=my_loss,optimizer=my_optimizer)
        tf.keras.utils.plot_model(self.model, to_file=self.output_directory+"_model.pdf", show_shapes=True)

        # print(self.model.summary())
        
        
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