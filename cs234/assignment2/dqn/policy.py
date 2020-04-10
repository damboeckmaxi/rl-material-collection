from abc import ABC
from typing import Generic, List, Tuple, Callable
import random
import numpy as np

import gym

from assignment2.dqn.dqn_types import ActionType, StateType

class DeterministicPolicy(ABC, Generic[StateType, ActionType]):
    def __call__(self, state: StateType) -> ActionType:
        raise NotImplementedError()


class StochasticPolicy(ABC, Generic[StateType, ActionType]):
    def __call__(self, state: StateType) -> List[Tuple[ActionType, float]]:
        """
        Stochastic policy interface.
        :param state: A generic typed state.
        :return: A list of tuples of the form (action, probability), representing possible actions and their
        probabilites under this policy. Actions which have probability 0 do not need to be contained in this list. The
        sum of probabilities should sum up to 1.
        """
        raise NotImplementedError()


class EpsilonGreedyPolicy(DeterministicPolicy, Generic[StateType, ActionType]):

    def __init__(self, q_function: Callable[[StateType], Tuple[ActionType, float]], epsilon: float, decay_steps: int = None, min_epsilon: float = 0):
        self._q_function = q_function
        self._epsilon = epsilon
        self._min_epsilon = min_epsilon

        if decay_steps is not None:
            self._decrement = (self._epsilon - min_epsilon) / decay_steps
        else:
            self._decrement = 0

    def __call__(self, state: StateType) -> ActionType:
        actions, q_values = zip(*self._q_function(state))
        if random.random() < self._epsilon:
            action = random.choice(actions)
        else:
            action = actions[np.argmax(q_values)]

        if self._epsilon > self._min_epsilon:
            self._epsilon -= self._decrement

        return action

def test1():

    q_table = {
        0: [(0, 0.1), (1, 0.1), (2, 0.2)],
        1: [(0, 0.1), (1, 0.2), (2, 0.1)],
        2: [(0, 0.2), (1, 0.0), (2, 0.2)]
    }

    policy = EpsilonGreedyPolicy[int, int](q_function=lambda s: q_table[s], epsilon=1.0, decay_steps=10)

    found_diff = False
    for i in range(100):
        rnd_act = policy(state=1)
        if rnd_act != 1:
            found_diff = True

    assert found_diff, "Test 1 failed."
    print("Test1: ok")

def test2():
    q_table = {
        0: [(0, 0.1), (1, 0.1), (2, 0.2)],
        1: [(0, 0.1), (1, 0.2), (2, 0.1)],
        2: [(0, 0.2), (1, 0.0), (2, 0.2)]
    }
    policy = EpsilonGreedyPolicy[int, int](q_function=lambda s: q_table[s], epsilon=1.0, decay_steps=10)
    for _ in range(5):
        _ = policy(0)
    assert round(policy._epsilon, 3) == 0.5, "Test 2 failed"
    print("Test2: ok")



if __name__ == "__main__":
    test1()
    test2()