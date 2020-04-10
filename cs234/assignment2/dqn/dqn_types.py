from typing import TypeVar, Tuple

StateType = TypeVar('StateType')
ActionType = TypeVar('ActionType')
RewardType = TypeVar('RewardType')

StateActionSequence = Tuple[StateType, ActionType, RewardType, StateType, bool]