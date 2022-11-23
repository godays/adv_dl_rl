from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np

from tic_tac_toe import TicTacToe


@dataclass(frozen=True)
class State:
    """Родительский класс для состояний среды"""
    pass


@dataclass(frozen=True)
class Action:
    """Возможный ход игрока"""
    i: int
    j: int

    def get_1d_index(self, n_cols: int) -> int:
        return self.i * n_cols + self.j


def empty_positions_to_actions(empty_positions: np.array) -> List[Action]:
    return [Action(*pos) for pos in empty_positions]


class Strategy(ABC):
    """Родительский класс для стратегий"""

    @abstractmethod
    def get_action(self, state: State, actions: List[Action]) -> Action:
        pass


class RandomStrategy(Strategy):
    def get_action(self, state: State, actions: List[Action]) -> Action:
        return np.random.choice(actions)

    def __repr__(self):
        return 'RandomStrategy'


class TicTacToeWrapper(ABC):
    """Оборачивает оригинальное окружение, чтобы возвращать состояния и действия в виде объектов State и Action"""

    @abstractmethod
    def __init__(self, env: TicTacToe):
        self.env = env

    @abstractmethod
    def reset(self) -> Tuple[State, List[Action]]:
        pass

    @abstractmethod
    def step(self, action: Action) -> Tuple[State, List[Action], int, bool]:
        pass
