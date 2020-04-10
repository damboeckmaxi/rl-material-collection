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
from assignment2.utils.general import get_logger


class SimpleReplayBuffer(ReplayBuffer[int, int, float]):

    def __init__(self, buffer_size: int):
        self.buffer = deque(maxlen=buffer_size)

    def store(self, sequence: StateActionSequence):
        self.buffer.append(sequence)

    def sample(self, n: int) -> List[StateActionSequence]:
        indices = np.random.choice(len(self.buffer), size=n)
        return [self.buffer[i] for i in indices]


class DQNAgent:

    def __init__(self, model: tf.keras.Model, config: Dict[str, str]):
        self.target_q_function = model
        self.q_function = tf.keras.models.clone_model(model)
        self.config = config

    @staticmethod
    def from_config(model: tf.keras.Model, file: str) -> DQNAgent:
        with open(file) as f:
            config = yaml.load(f)
            return DQNAgent(model=model, config=config)

    def act(self, state: np.ndarray) -> int:
        q_values = self.target_q_function.predict(np.array([state,]))
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
        train_interval = int(self.config['train_interval'])
        replay_buffer = self.configure_replay_buffer()
        policy = self.configure_policy()

        t = 0
        while t < steps:
            total_reward = 0
            state = env.reset()
            while True:
                t += 1

                action = policy(state=state)
                next_state, reward, done, info = env.step(action)
                replay_buffer.store((state, action, reward, next_state, done))
                state = next_state

                if t > t_start_training:
                    losses = self.fit_model(replay_buffer=replay_buffer, gamma=gamma)
                    if t % train_interval == 0:
                        self.target_q_function = tf.keras.models.clone_model(self.q_function)

                if t >= steps or done:
                    break

    def get_action_values(self, state: np.ndarray) -> List[Tuple[int, float]]:
        q_values = self.q_function.predict(np.array([state,]))[0]
        return list([(i, q) for i, q in enumerate(q_values)])

    def fit_model(self, replay_buffer: ReplayBuffer, gamma: float):
        batch_size = int(self.config['batch_size'])
        learning_rate = float(self.config['learning_rate'])

        replay_batch = replay_buffer.sample(n=batch_size)

        states = np.array([s for s, _, _, _, _ in replay_batch])

        y = self.q_function.predict(states)

        for i in range(batch_size):
            state, action, reward, next_state, done = replay_batch[i]
            if done:
                y[i][action] = reward
            else:
                q_values = self.target_q_function.predict(np.array([next_state,]))
                y[i][action] = reward + gamma * np.max(q_values)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=10.0)
        self.q_function.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'acc'])

        losses = self.q_function.fit(states, y, batch_size=batch_size)
        return losses

    @property
    def model(self):
        return tf.keras.models.clone_model(self.target_q_function)