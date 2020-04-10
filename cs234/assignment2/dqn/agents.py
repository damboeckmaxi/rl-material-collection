from __future__ import annotations

import random
from collections import deque
from logging import Logger

import gym
import numpy as np
import yaml
import os
import tensorflow as tf
from abc import ABC
from typing import Generic, Dict, List, Tuple

from assignment2.dqn.dqn_types import StateActionSequence
from assignment2.dqn.experience_replay import ReplayBuffer
from assignment2.dqn.policy import EpsilonGreedyPolicy
from assignment2.utils.general import get_logger, Progbar


class SimpleReplayBuffer(ReplayBuffer[int, int, float]):

    def __init__(self, buffer_size: int):
        self.buffer = deque(maxlen=buffer_size)

    def store(self, sequence: StateActionSequence):
        self.buffer.append(sequence)

    def sample(self, n: int) -> List[StateActionSequence]:
        indices = list(range(len(self.buffer)))
        random.shuffle(indices)
        sample = [self.buffer[i] for i in indices[:n]]
        return sample


class DQNAgent:

    def __init__(self, model: tf.keras.Model, config: Dict[str, str]):
        self.progress = Progbar(5000)
        self.target_q_function = model
        self.q_function = tf.keras.models.clone_model(model)

        learning_rate = float(config['learning_rate'])
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=10.0)
        self.q_function.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'acc'])

        self.config = config
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='../results', histogram_freq=1)

    @staticmethod
    def from_config(model: tf.keras.Model, file: str) -> DQNAgent:
        with open(file) as f:
            config = yaml.load(f)
            return DQNAgent(model=model, config=config)

    def act(self, state: np.ndarray) -> int:
        q_values = self.q_function.predict(np.array([state,]))
        return np.argmax(q_values, axis=-1)[0]

    def configure_policy(self) -> EpsilonGreedyPolicy:
        return EpsilonGreedyPolicy[np.ndarray, int](
            q_function=self.get_action_values,
            epsilon=float(self.config['epsilon']),
            min_epsilon=float(self.config['epsilon_min']),
            decay_steps=int(self.config['epsilon_decay'])
        )

    def configure_replay_buffer(self) -> SimpleReplayBuffer:
        return SimpleReplayBuffer(int(self.config['buffer_size']))


    def learn(self, env: gym.Env, steps: int, gamma: float):
        t_start_training = int(self.config['t_start_training'])
        target_update_interval = int(self.config['target_update_interval'])
        replay_buffer = self.configure_replay_buffer()
        policy = self.configure_policy()

        t = 0
        while t < steps:
            state = env.reset()
            while True:
                t += 1
                self.progress.update(t)

                action = policy(state=state)
                next_state, reward, done, _ = env.step(action)
                replay_buffer.store((state, action, reward, next_state, done))
                state = next_state

                if t > t_start_training:
                    self.fit_model(replay_buffer=replay_buffer, gamma=gamma)
                    if t % target_update_interval == 0:
                        self.target_q_function.set_weights(self.q_function.get_weights())

                if t >= steps or done:
                    break

    def get_action_values(self, state: np.ndarray) -> List[Tuple[int, float]]:
        q_values = self.q_function.predict(np.array([state,]))[0]
        return list([(i, q) for i, q in enumerate(q_values)])

    def fit_model(self, replay_buffer: ReplayBuffer, gamma: float):
        batch_size = int(self.config['batch_size'])

        replay_batch = replay_buffer.sample(n=batch_size)
        states = np.array([s for s, _, _, _, _ in replay_batch])
        next_states = np.array([s for _, _, _, s, _ in replay_batch])

        y = self.q_function.predict(states)
        next_q_values = self.target_q_function.predict(next_states)

        for i in range(batch_size):
            state, action, reward, next_state, done = replay_batch[i]
            if done:
                y[i][action] = reward
            else:
                y[i][action] = reward + gamma * np.max(next_q_values[i])

        self.q_function.train_on_batch(states, y)

    @property
    def model(self):
        return tf.keras.models.clone_model(self.target_q_function)