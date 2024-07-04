import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.initializers import RandomNormal

def create_q_network(state_size, action_size, repeat_factor=15):
    inputs = layers.Input(shape=state_size)

    # Repeat the input tensor to increase its dimensionality
    # Repeat the input tensor to increase its dimensionality
    # Expand dimensions to add a channel dimension and repeat the input tensor to increase its dimensionality
    expanded_inputs = tf.expand_dims(inputs, -1)  # Shape becomes (None, 9, 1)
    repeated_inputs = tf.tile(expanded_inputs, [1, repeat_factor, 1])  # Repeating along the time dimension, shape becomes (None, 9 * repeat_factor, 1)



    # 1st CNN layer with max-pooling
    conv1 = layers.Convolution1D(256, 7, kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), 
                                 bias_initializer=RandomNormal(mean=0.0, stddev=0.05), activation='relu')(repeated_inputs)
    pool1 = layers.MaxPooling1D(pool_size=3)(conv1)

    # 2nd CNN layer with max-pooling
    conv2 = layers.Convolution1D(256, 7, kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), 
                                 bias_initializer=RandomNormal(mean=0.0, stddev=0.05), activation='relu')(pool1)
    pool2 = layers.MaxPooling1D(pool_size=3)(conv2)

    # 3rd CNN layer without max-pooling
    conv3 = layers.Convolution1D(256, 3, kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), 
                                 bias_initializer=RandomNormal(mean=0.0, stddev=0.05), activation='relu')(pool2)

    # 4th CNN layer without max-pooling
    conv4 = layers.Convolution1D(256, 3, kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), 
                                 bias_initializer=RandomNormal(mean=0.0, stddev=0.05), activation='relu')(conv3)

    # 5th CNN layer without max-pooling
    conv5 = layers.Convolution1D(256, 3, kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), 
                                 bias_initializer=RandomNormal(mean=0.0, stddev=0.05), activation='relu')(conv4)

    # 6th CNN layer with max-pooling
    conv6 = layers.Convolution1D(256, 3, kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), 
                                 bias_initializer=RandomNormal(mean=0.0, stddev=0.05), activation='relu')(conv5)
    pool6 = layers.MaxPooling1D(pool_size=3)(conv6)

    # Reshaping to 1D array for further layers
    flat = layers.Flatten()(pool6)

    # 1st fully connected layer with dropout
    dense1 = layers.Dense(1024, kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), 
                          bias_initializer=RandomNormal(mean=0.0, stddev=0.05), activation='relu')(flat)
    dropout1 = layers.Dropout(0.5)(dense1)

    # 2nd fully connected layer with dropout
    dense2 = layers.Dense(1024, kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), 
                          bias_initializer=RandomNormal(mean=0.0, stddev=0.05), activation='relu')(dropout1)
    dropout2 = layers.Dropout(0.5)(dense2)

    # Output layer with linear activation
    outputs = layers.Dense(action_size, kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), 
                           bias_initializer=RandomNormal(mean=0.0, stddev=0.05), activation='linear')(dropout2)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model

