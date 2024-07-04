

# Define the model
"""
Title: Actor Critic Method
Author: [Apoorv Nandan](https://twitter.com/NandanApoorv)
Date created: 2020/05/13
Last modified: 2020/05/13
Description: Implement Actor Critic Method in CartPole environment.
"""
"""
## Introduction
This script shows an implementation of Actor Critic method on CartPole-V0 environment.
### Actor Critic Method
As an agent takes actions and moves through an environment, it learns to map
the observed state of the environment to two possible outputs:
1. Recommended action: A probability value for each action in the action space.
   The part of the agent responsible for this output is called the **actor**.
2. Estimated rewards in the future: Sum of all rewards it expects to receive in the
   future. The part of the agent responsible for this output is the **critic**.
Agent and Critic learn to perform their tasks, such that the recommended actions
from the actor maximize the rewards.
### CartPole-V0
A pole is attached to a cart placed on a frictionless track. The agent has to apply
force to move the cart. It is rewarded for every time step the pole
remains upright. The agent, therefore, must learn to keep the pole from falling over.
### References
- [CartPole](http://www.derongliu.org/adp/adp-cdrom/Barto1983.pdf)
- [Actor Critic Method](https://hal.inria.fr/hal-00840470/document)
"""
"""
## Setup
"""
import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#from custom_environment2 import NASEnvironment

from tsp_env import TSPEnv


distances = [
    [0, 2, 2, 5, 9, 3],
    [2, 0, 4, 6, 7, 8],
    [2, 4, 0, 8, 6, 3],
    [5, 6, 8, 0, 4, 9],
    [9, 7, 6, 4, 0, 10],
    [3, 8, 3, 9, 10, 0]
]

env = TSPEnv(distances)

# Example random agent


sequence_len = len(distances)+1
maxlen = sequence_len
num_inputs = maxlen
num_actions = len(distances)
num_hidden = 128


# Configuration parameters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
max_steps_per_episode = 2 #10000
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0


initial_epsilon = 0.1
decay_factor = 0.99

eps_action = initial_epsilon
eps_position = initial_epsilon


#num_inputs = 4
#num_actions = 3
#num_hidden = 128
#############################################################################################################

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
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
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



model_action = get_vqvae(maxlen, num_actions)

"""
## Train
"""

#optimizer = keras.optimizers.Adam(learning_rate=0.01)
optimizer = keras.optimizers.Adam()

huber_loss = keras.losses.Huber()

action_probs_history = []
position_probs_history = []
critic_value_history = []
critic_pos_value_history = []
action_pos_probs_history = []
critic_value_pos_history = []

action_probs_history_next = []
position_probs_history_next = []
critic_value_history_next = []
critic_pos_value_history_next = []

action_pos_probs_history_next = []
critic_value_pos_history_next = []


rewards_history = []
running_reward = 0
episode_count = 0

rewards2save = []

# Initialize parameters
epsilon = 1.0          # Initial epsilon
epsilon_min = 0.01     # Minimum epsilon
epsilon_decay = 0.995  # Decay rate
alpha = 0.1            # Learning rate
gamma = 0.99           # Discount factor



state = env.reset()
episode_reward = 0
# state = tf.convert_to_tensor(state)
# state = tf.expand_dims(state, 0)


#state = np.array([0, 1, 8, 2, 6, 5, 8, 3, 6, 4, 9, 1, 1, 1, 1])

done = False

while True:  # Run until solved
    state = env.reset()
    episode_reward = 0
    with tf.GradientTape() as tape:
        while not done:
        
        
            # env.render(); Adding this line would show the attempts
            # of the agent in a pop up window.

            # if np.random.rand() < epsilon:
            #     action = np.random.randint(num_actions)  # Exploration 
            # else:

            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            # Predict action probabilities and estimated future rewards
            # from environment state
            action_probs, critic_value = model_action(state)
            critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distribution
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            action_probs_history.append(tf.math.log(action_probs[0, action]))

            #calculate advantage
            action_probs_next, critic_value_next = model_action.predict(state)#(tf.convert_to_tensor(tf.expand_dims(state, 0)))
            #store values
            action_probs_history_next.append(action_probs_next)
            critic_value_history_next.append(critic_value_next)

            # # Decay epsilon
            # if epsilon > epsilon_min:
            #     epsilon *= epsilon_decay

            print('action=', action)
            state, reward, done, _ = env.step(action)
            env.render()
            rewards_history.append(reward)
            # rewards2save.append(reward)
            episode_reward += reward

            # if done:
            #     break

        # Update running reward to check condition for solving
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()


        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, \
                      returns,\
                      action_probs_history_next, critic_value_history_next)
        actor_losses = []
        actor_pos_losses = []
        critic_losses = []
        critic_pos_losses = []
        for log_prob, value, ret, log_prob_next, value_next in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss


            # # The critic must be updated so that it predicts a better estimate of
            # # the future rewards.
            # critic_losses.append(
            #     huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            # )

            # critic_pos_losses.append(
            #     huber_loss_pos(tf.expand_dims(value_pos, 0), tf.expand_dims(ret, 0))
            # )

            # Calculate advantage
            advantage = reward + gamma * value_next - value
            # Update Critic
            critic_losses.append(tf.math.pow(advantage,2))

        # Backpropagation
        if actor_losses and critic_losses:
            loss_value = sum(actor_losses) + sum(critic_losses)
            grads = tape.gradient(loss_value, model_action.trainable_variables)
            grads = [tf.convert_to_tensor(g) for g in grads]
            optimizer.apply_gradients(zip(grads, model_action.trainable_variables))


                
        # loss_value = sum(actor_losses) + sum(critic_losses) + sum(actor_pos_losses) + sum(critic_pos_losses)
        # grads = tape.gradient(loss_value, model_action.trainable_variables+model_position.trainable_variables)
        # optimizer.apply_gradients(zip(grads, model_action.trainable_variables+model_position.trainable_variables))

    # Clear the loss and reward history
    action_probs_history.clear()
    critic_value_history.clear()

    action_pos_probs_history.clear()
    critic_value_pos_history.clear()

    action_probs_history_next.clear()
    critic_value_history_next.clear()

    action_pos_probs_history_next.clear()
    critic_value_pos_history_next.clear()

    rewards_history.clear()

    # # Log details
    # episode_count += 1
    # if episode_count % 10 == 0:
    #     template = "running reward: {:.2f} at episode {}"
    #     print(template.format(running_reward, episode_count))

    # if running_reward > 195:  # Condition to consider the task solved
    #     print("Solved at episode {}!".format(episode_count))
    #     break

    # if done:
    #     break
# import pickle

# # File path to save the array
# file_path = "rewards_array_data.pkl"

# # Dumping the array into a file using pickle
# with open(file_path, 'wb') as f:
#     pickle.dump(rewards2save, f)

# print("Array saved successfully!")

