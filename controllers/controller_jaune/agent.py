import tensorflow as tf
import numpy as np
from network import QNetwork

class QAgent:
    def __init__(self, input_size, output_size, hidden_size, learning_rate, gamma):
        self.q_network = QNetwork(input_size, output_size, hidden_size)
        self.target_network = QNetwork(input_size, output_size, hidden_size)
        self.target_network.set_weights(self.q_network.get_weights())
        self.buffer = []
        self.batch_size = 64
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.gamma = gamma

    def get_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            speed = np.random.uniform(0, 28)
            angle = np.random.uniform(-16, 16)
            return speed, angle
        else:
            state = np.expand_dims(state, axis=0)
            action = self.q_network(state)
            return action[0]

    def train(self):
        if len(self.buffer) < self.batch_size:
            return

        samples = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in samples])

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        with tf.GradientTape() as tape:
            target_actions = self.target_network(next_states)
            target_q_values = rewards + self.gamma * (1 - dones) * target_actions

            current_actions = self.q_network(states)
            current_q_values = current_actions

            loss = tf.reduce_mean(tf.losses.mean_squared_error(target_q_values, current_q_values))

        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))

        return loss.numpy()

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())