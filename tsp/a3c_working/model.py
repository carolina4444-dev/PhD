import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras as keras

# class ActorCriticModel(tf.keras.Model):
#     def __init__(self, num_actions):
#         super(ActorCriticModel, self).__init__()
#         self.common = layers.Dense(128, activation="relu")
#         self.actor = layers.Dense(num_actions, activation="softmax")
#         self.critic = layers.Dense(1)

#     def call(self, inputs):
#         x = self.common(inputs)
#         return self.actor(x), self.critic(x)

############################################################
    

class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = (
            beta  # This parameter is best kept between [0.25, 2] as per the paper.
        )

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        # encoding_indices = self.get_code_indices(flattened)
        # encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        # quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        # quantized = tf.reshape(quantized, input_shape)

        encoding_indices = self.get_code_indices(flattened)
        #encoding_indices = tf.cast(encoding_indices, dtype=tf.float32)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)

        # # Calculate vector quantization loss and add that to the layer. You can learn more
        # # about adding losses to different layers here:
        # # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # # the original paper to get a handle on the formulation of the loss function.
        # commitment_loss = self.beta * tf.reduce_mean(
        #     (tf.stop_gradient(quantized) - x) ** 2
        # )
        # codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        # self.add_loss(commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized
    
    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs**2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings**2, axis=0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices
    

    # def get_code_indices(self, flattened_inputs):
    #     # Calculate L2-normalized distance between the inputs and the codes.
    #     similarity = tf.matmul(flattened_inputs, self.embeddings)
    #     distances = (
    #         tf.reduce_sum(flattened_inputs**2, axis=1, keepdims=True)
    #         + tf.reduce_sum(self.embeddings**2, axis=0)
    #         - 2 * similarity
    #     )

    #     # Derive the indices for minimum distances.
    #     encoding_indices = tf.argmin(distances, axis=1)
    #     return encoding_indices


def get_encoder(inputs, head_size=256, num_heads=4, ff_dim=4, dropout=0.25):
    # Attention and Normalization
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    encoder_outputs = x + res

    return encoder_outputs #keras.Model(inputs, encoder_outputs, name="encoder")


def get_decoder(input_shape, mlp_units=[128], mlp_dropout=0.4):
    #inputs = keras.Input(shape=build_encoder_model(input_shape).output.shape[1:])
    inputs = keras.Input(shape=(input_shape))
    x = inputs
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    #outputs = layers.Dense(output_shape, activation="relu")(x)

    critic_outputs = layers.Dense(1, activation="relu")(x)
    # x = tf.reshape(x, (-1, maxlen, num_actions))
    # outputs = tf.argmax(x, axis=2)
    return keras.Model(inputs, critic_outputs)


def build_encoder_model(input_shape, output_dim, head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=4, mlp_units=[128], dropout=0, mlp_dropout=0):
    inputs = layers.Input(shape=input_shape)
    x = keras.backend.expand_dims(inputs, axis=-1)
    for _ in range(num_transformer_blocks):
        x = get_encoder(x, head_size, num_heads, ff_dim, dropout)

    #output = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    output = x

    encoder_outputs = layers.Flatten()(output)
    action = layers.Dense(output_dim)(encoder_outputs)

    critic = layers.Dense(1)(encoder_outputs)

    return keras.Model(inputs, [action, critic])

#######transformer position############################

def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    """
    Mask the upper half of the dot product matrix in self attention.
    This prevents flow of information from future tokens to current token.
    1's in the lower triangle, counting from the lower right corner.
    """
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest
    mask = tf.cast(m, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
    )
    return tf.tile(mask, mult)


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        
        self.att = layers.MultiHeadAttention(num_heads, embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        super(TransformerBlock, self).__init__()

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        if batch_size is None:
            batch_size = 1
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        attention_output = self.att(inputs, inputs, attention_mask=causal_mask)
        attention_output = self.dropout1(attention_output)
        out1 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


"""
## Implement an embedding layer
Create two seperate embedding layers: one for tokens and one for token index
(positions).
"""


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
        #return positions


def get_vqvae(input_dim, output_dim):
    embed_dim = 32  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer
    vocab_size = 20000  # Only consider the top 20k words

    inputs = layers.Input(shape=(input_dim,))
    embedding_layer = TokenAndPositionEmbedding(input_dim, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    action = layers.Dense(output_dim, activation="softmax")(x)
    critic = layers.Dense(1, activation="relu")(x)

    model = keras.Model(inputs, outputs=[action, critic], name="vq_vae")

    return model




