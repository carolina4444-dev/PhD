import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, AveragePooling1D, concatenate, Add, Dense, Embedding, GlobalMaxPooling1D
from tensorflow.keras.models import Model

def nas_network_builder(sequence, input_shape, num_classes):
    """
    Builds a Keras model based on the provided NAS sequence.

    Args:
        sequence (list of int): The integer sequence encoding the network architecture.
        input_shape (tuple): The shape of the input data (excluding batch size).
        num_classes (int): The number of output classes.

    Returns:
        keras.Model: The constructed Keras model.
    """
    input_layer = Input(shape=input_shape)
    x = Embedding(input_dim=5000, output_dim=128, input_length=input_shape[0])(input_layer)
    
    # Dictionary to store intermediate layers for skip/residual connections
    layers = {0: x}
    current_layer = x
    index = 0

    # Predefined configurations
    conv_filters = 64
    conv_kernel_size = 3
    pool_size = 2

    while index < len(sequence):
        layer_type = sequence[index]
        index += 1
        
        if layer_type == 0:  # Conv1D layer
            current_layer = Conv1D(filters=conv_filters, kernel_size=conv_kernel_size, activation='relu', padding='same')(current_layer)
        elif layer_type == 1:  # MaxPooling1D layer
            current_layer = MaxPooling1D(pool_size=pool_size, padding='same')(current_layer)
        elif layer_type == 2:  # AveragePooling1D layer
            current_layer = AveragePooling1D(pool_size=pool_size, padding='same')(current_layer)
        elif layer_type == 3:  # Skip Connection
            target_layer_index = sequence[index]
            current_layer = concatenate([current_layer, layers[target_layer_index]], axis=-1)
            index += 1
        elif layer_type == 4:  # Residual Connection
            target_layer_index = sequence[index]
            current_layer = Add()([current_layer, layers[target_layer_index]])
            index += 1
        
        # Store the current layer in the dictionary
        layers[len(layers)] = current_layer

    # Final layers for classification
    current_layer = GlobalMaxPooling1D()(current_layer)
    current_layer = Dense(128, activation='relu')(current_layer)
    output_layer = Dense(num_classes, activation='softmax')(current_layer)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Example usage:
sequence = [0, 0, 1, 0, 2, 0, 4, 2, 3, 1]  # Example sequence
input_shape = (100,)  # Example input shape for text data
num_classes = 10  # Number of output classes

model = nas_network_builder(sequence, input_shape, num_classes)
model.summary()
