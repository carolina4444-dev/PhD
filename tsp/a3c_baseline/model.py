import tensorflow as tf
from tensorflow.keras import layers

class ActorCriticModel(tf.keras.Model):
    def __init__(self, num_actions):
        super(ActorCriticModel, self).__init__()
        self.common = layers.Dense(128, activation="relu")
        self.actor = layers.Dense(num_actions, activation="softmax")
        self.critic = layers.Dense(1)

    def call(self, inputs):
        x = self.common(inputs)
        return self.actor(x), self.critic(x)


