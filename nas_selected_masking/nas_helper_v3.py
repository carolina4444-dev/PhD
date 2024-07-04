import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.initializers import RandomNormal

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Flatten, Dense, Input, Concatenate, Dropout
from tensorflow.keras.models import Model
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


# Replace 'your_dataset.csv' with the actual path to your CSV file
csv_file = 'train_set.csv'

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file, encoding='utf-8')

# # Display the DataFrame
# print(df)

column_name = 'motivo contacto'
input = df[column_name].tolist()

column_name = 'ultimo algoritmo'
output = df[column_name].tolist()

import nltk
from tensorflow.keras.preprocessing.text import Tokenizer

# Get the list of Portuguese stopwords
stop_words = nltk.corpus.stopwords.words('portuguese')

# Function to preprocess and tokenize text while filtering out stopwords
def preprocess_text(text):
    
    # Remove stopwords
    filtered_words = [word for word in text if word.lower() not in stop_words]
    
    return ' '.join(filtered_words)

# Preprocess the text data
preprocessed_texts = [preprocess_text(text) for text in input]


#output

#categorize output
from sklearn import preprocessing


le = preprocessing.OneHotEncoder()
categorical_labels = le.fit_transform(np.reshape(output, (-1,1))).toarray()

total_elems = len(le.categories_[0])


from sklearn.model_selection import train_test_split
import numpy as np


def one_hot_encode_char(char, char_dict):
    # Create a zero vector of the length equal to the dictionary size
    one_hot_vector = np.zeros(len(char_dict))
    # Set the position corresponding to the char to 1
    one_hot_vector[char_dict[char]] = 1
    return one_hot_vector

def one_hot_encode_sentences(sentences, char_dict, max_length):
    # Initialize the encoded data array
    encoded_data = np.zeros((len(sentences), max_length, len(char_dict)), dtype=np.float32)
    
    # Encode each sentence
    for i, sentence in enumerate(sentences):
        for j, char in enumerate(sentence):
            if char in char_dict:
                encoded_data[i, j] = one_hot_encode_char(char, char_dict)
    return encoded_data

# Example usage
sentences = preprocessed_texts
char_dict = {ch: idx for idx, ch in enumerate(sorted(set(''.join(sentences))))}
max_length = max(len(sentence) for sentence in sentences)

encoded_sentences = one_hot_encode_sentences(sentences, char_dict, max_length)
print("Encoded sentences shape:", encoded_sentences.shape)
print("Encoded data for 'hello':\n", encoded_sentences[0])


x_train, x_test, y_train, y_test = train_test_split(encoded_sentences, categorical_labels, test_size=0.2, random_state=42)



def inception_module(x, filters):
    # Each branch of the Inception module
    branch1x1 = layers.Conv2D(filters[0], 1, padding='same', activation='relu')(x)

    branch3x3 = layers.Conv2D(filters[1], 1, padding='same', activation='relu')(x)
    branch3x3 = layers.Conv2D(filters[2], 3, padding='same', activation='relu')(branch3x3)

    branch5x5 = layers.Conv2D(filters[3], 1, padding='same', activation='relu')(x)
    branch5x5 = layers.Conv2D(filters[4], 5, padding='same', activation='relu')(branch5x5)

    branch_pool = layers.MaxPooling2D(3, strides=1, padding='same')(x)
    branch_pool = layers.Conv2D(filters[5], 1, padding='same', activation='relu')(branch_pool)

    # Concatenate all the branches
    output = layers.Concatenate()([branch1x1, branch3x3, branch5x5, branch_pool])
    return output

# def googlenet():
#     inputs = Input(shape=((2, 256)))
    
#     x = Conv1D(64, 7, strides=2, padding='same', activation='relu')(inputs)
#     x = MaxPooling1D(3, strides=2, padding='same')(x)
#     x = Conv1D(192, 3, padding='same', activation='relu')(x)
#     x = MaxPooling1D(3, strides=2, padding='same')(x)

#     # Inception modules
#     x = inception_module(x, [64, 96, 128, 16, 32, 32])
#     x = inception_module(x, [128, 128, 192, 32, 96, 64])
#     x = MaxPooling1D(3, strides=2, padding='same')(x)

#     x = inception_module(x, [192, 96, 208, 16, 48, 64])
#     x = inception_module(x, [160, 112, 224, 24, 64, 64])
#     x = inception_module(x, [128, 128, 256, 24, 64, 64])
#     x = inception_module(x, [112, 144, 288, 32, 64, 64])
#     x = inception_module(x, [256, 160, 320, 32, 128, 128])
#     x = MaxPooling1D(3, strides=2, padding='same')(x)

#     x = inception_module(x, [256, 160, 320, 32, 128, 128])
#     x = inception_module(x, [384, 192, 384, 48, 128, 128])

#     x = AveragePooling1D(7, strides=1)(x)
#     x = Dropout(0.4)(x)
#     x = Flatten()(x)
#     x = Dense(1000, activation='softmax')(x)

#     model = Model(inputs, x)
#     return model

from keras.models import Model
from keras.layers import Input, Conv1D, Activation, Add, GlobalMaxPooling1D, Dense
from keras.backend import expand_dims

def residual_block(x, filters, kernel_size, dilation_rate):
    # Convolutional layers
    c1 = Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='same')(x)
    a1 = Activation('relu')(c1)
    c2 = Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='same')(a1)

    # Residual connection
    if x.shape[-1] == filters:
        shortcut = x
    else:
        shortcut = Conv1D(filters=filters, kernel_size=1, padding='same')(x)
    r = Add()([c2, shortcut])
    return Activation('relu')(r)



#def create_q_model(input_shape, num_filters, kernel_size, strides, dilation_rate, output_dim):

def create_q_model(input_shape, num_filters, kernel_size, strides, dilation_rate, output_dim):

    inputs = Input(shape=input_shape)

    
    conv1 = layers.Convolution1D(256,7,kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05),activation='relu')(inputs)#sequence_one_hot
    pool1 = layers.MaxPooling1D(pool_size=3)(conv1)

    # 2nd CNN layer with max-pooling
    conv2 = layers.Convolution1D(256,7,kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05),activation='relu')(pool1)
    pool2 = layers.MaxPooling1D(pool_size=3)(conv2)

    # 3rd CNN layer without max-pooling
    conv3 = layers.Convolution1D(256,3,kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05),activation='relu')(pool2)

    # 4th CNN layer without max-pooling
    conv4 = layers.Convolution1D(256,3,kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05),activation='relu')(conv3)

    # 5th CNN layer without max-pooling
    conv5 = layers.Convolution1D(256,3,kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05),activation='relu')(conv4)

    # 6th CNN layer with max-pooling
    conv6 = layers.Convolution1D(256,3,kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05),activation='relu')(conv5)
    pool6 = layers.MaxPooling1D(pool_size=3)(conv6)


    # Fully connected layers (optional)
    # x = layers.Dense(units=..., activation='relu')(x)
    # Add more dense layers if needed

    # outputs = layers.Dense(units=1, activation='sigmoid')(x)
    x = expand_dims(pool6, axis=-1)

    # print(conv3.shape)
    # x = conv3
    # x = layers.Reshape((512, 1))(conv3)
    # x = layers.BatchNormalization()(x)

    #classification

    x = layers.Conv2D(64, 7, strides=2, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    x = layers.Conv2D(192, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    #x = layers.BatchNormalization()(x)

    # Inception modules
    x = inception_module(x, [64, 96, 128, 16, 32, 32])
    # x = inception_module(x, [128, 128, 192, 32, 96, 64])
    # x = MaxPooling1D(3, strides=2, padding='same')(x)

    # x = inception_module(x, [192, 96, 208, 16, 48, 64])
    # x = inception_module(x, [160, 112, 224, 24, 64, 64])
    # x = inception_module(x, [128, 128, 256, 24, 64, 64])
    # x = inception_module(x, [112, 144, 288, 32, 64, 64])
    # x = inception_module(x, [256, 160, 320, 32, 128, 128])
    # x = MaxPooling1D(3, strides=2, padding='same')(x)

    # x = inception_module(x, [256, 160, 320, 32, 128, 128])
    # x = inception_module(x, [384, 192, 384, 48, 128, 128])

    #x = AveragePooling1D(7, strides=1)(x)
    #x = Dropout(0.4)(x)
    x = Flatten()(x)

    # Dense layers
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(output_dim, activation='softmax')(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


output_dim = 52
input_shape = x_train.shape[1:]

num_filters = [64, 128, 256]  # Number of filters for each temporal convolutional layer
kernel_size = 3  # Kernel size for convolutional layers
strides = 1  # Strides for convolutional layers
dilation_rate = [1, 2, 4]  # Dilation rate for each temporal convolutional layer


# # Example usage
# num_filters = 64
# kernel_size = 3
# dilation_rates = [1, 2, 4, 8]  # Example dilation rates for the residual blocks




model = create_q_model(input_shape, num_filters, kernel_size, strides, dilation_rate, output_dim)


# # # Create the GoogLeNet model
# model = googlenet()

# Model summary
model.summary()


model.compile(
    optimizer='adam',                   # Optimizer
    loss='categorical_crossentropy',    # Loss function for multi-class classification
    metrics=['accuracy']                # List of metrics to monitor
)

# Parameters
batch_size = 32
epochs = 75

# Fit the model
history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test)
)


test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')
