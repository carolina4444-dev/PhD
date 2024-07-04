import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, initializers



def create_q_network(state_size, action_size):
    inputs = layers.Input(shape=(state_size, 1))

    # Inception module with parallel Conv1D layers
    tower_1 = layers.Conv1D(64, 1, padding='same', activation='relu', 
                            kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05))(inputs)
    
    tower_2 = layers.Conv1D(64, 1, padding='same', activation='relu', 
                            kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05))(inputs)
    tower_2 = layers.Conv1D(64, 3, padding='same', activation='relu', 
                            kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05))(tower_2)
    
    tower_3 = layers.Conv1D(64, 1, padding='same', activation='relu', 
                            kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05))(inputs)
    tower_3 = layers.Conv1D(64, 5, padding='same', activation='relu', 
                            kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05))(tower_3)
    
    tower_4 = layers.MaxPooling1D(3, strides=1, padding='same')(inputs)
    tower_4 = layers.Conv1D(64, 1, padding='same', activation='relu', 
                            kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05))(tower_4)
    
    # Concatenate the outputs of the inception module
    concatenated = layers.Concatenate(axis=-1)([tower_1, tower_2, tower_3, tower_4])
    
    # Additional Conv1D layers
    conv = layers.Conv1D(256, 3, padding='same', activation='relu', 
                         kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05))(concatenated)
    pool = layers.MaxPooling1D(pool_size=2)(conv)
    
    # Flatten and fully connected layers
    flat = layers.Flatten()(pool)
    dense1 = layers.Dense(128, activation='relu', 
                          kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05))(flat)
    dropout1 = layers.Dropout(0.5)(dense1)
    
    # Output layer
    outputs = layers.Dense(action_size, 
                           kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05))(dropout1)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')
    return model