from abc import ABC
from typing import Generic, List

from assignment2.dqn.dqn_types import StateType, ActionType, RewardType, StateActionSequence


class ReplayBuffer(ABC, Generic[StateType, ActionType, RewardType]):
    def store(self, sequence: StateActionSequence):
        pass

    def sample(self, n: int) -> List[StateActionSequence]:
        pass