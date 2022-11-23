from dataclasses import dataclass, field
from typing import Tuple, List, Any, Optional, Iterable

import numpy as np

from contracts import Action, State, Strategy, TicTacToeWrapper, empty_positions_to_actions
from table_functions import TableFunction
from tic_tac_toe import TicTacToe


@dataclass(frozen=True)
class Afterstate(State):
    board_hash: str  # Строковое представление состояния доски


@dataclass(frozen=True)
class TableState(State):
    """Состояние для табличных методов обучения"""
    current_turn: int  # Чей сейчас ход
    board_hash: str  # Строковое представление состояния доски
    available_actions: List[Action] = field(hash=False, repr=False)  # Список действий, доступных в данном состоянии

    def get_afterstate(self, action: Action) -> Afterstate:
        board_size = int(np.sqrt(len(self.board_hash)))
        action_index = action.get_1d_index(board_size)
        afterstate_hash = (self.board_hash[:action_index]
                           + str(self.current_turn + 1)
                           + self.board_hash[action_index + 1:])
        return Afterstate(afterstate_hash)


class TicTacToeTableWrapper(TicTacToeWrapper):
    """Оборачивает оригинальное окружение, чтобы возвращать состояния и действия в виде объектов TableState и Action,
    пригодных для табличных методов"""

    def __init__(self, env: TicTacToe):
        super().__init__(env)

    def reset(self) -> Tuple[TableState, List[Action]]:
        self.env.reset()
        env_state, reward, done, _ = self.env.reset()
        board_hash, empty_positions, current_turn = env_state
        actions = empty_positions_to_actions(empty_positions)
        state = TableState(current_turn, board_hash, actions)
        return state, actions

    def step(self, action: Action) -> Tuple[TableState, List[Action], int, bool]:
        env_state, crosses_reward, done, _ = self.env.step((action.i, action.j))
        board_hash, empty_positions, current_turn = env_state
        actions = empty_positions_to_actions(empty_positions)
        state = TableState(current_turn, board_hash, actions)
        #  crosses_reward - реворд для крестиков
        #  реворд для ноликов == -1*crosses_reward
        return state, actions, crosses_reward, done


class AfterstatesTableFunctionWrapper(TableFunction):
    """Проксирует обращения к TableFunction, заменяя пару (состояние, действие)
    на afterstate"""

    def __init__(self, function: TableFunction):
        self.function = function

    def __getattr__(self, attr):
        if attr.startswith('__') and attr.endswith('__'):
            raise AttributeError
        return getattr(self.function, attr)

    def update(self, key: Tuple[TableState, Action], value: float) -> None:
        state, action = key
        self.function.update(state.get_afterstate(action), value)

    def merge(self, functions: List[TableFunction]) -> None:
        raise NotImplementedError

    def get(self, key: Tuple[TableState, Action]) -> float:
        state, action = key
        return self.function.get(state.get_afterstate(action))

    def items(self) -> Iterable[Tuple[Any, float]]:
        raise NotImplementedError

    def keys(self) -> Iterable[Any]:
        raise NotImplementedError

    def show(self, top: Optional[int] = None) -> str:
        return self.function.show(top)


class EpsilonGreedyStrategy(Strategy):
    def __init__(self, epsilon: float, Q: TableFunction):
        self.epsilon = epsilon
        self.Q = Q

    def _get_best_action(self, state: TableState, actions: List[Action]) -> Action:
        """Возвращает действие, максимизирующее Q(state, a)"""
        action_values = [self.Q[(state, action)] for action in actions]
        return actions[np.argmax(action_values)]

    def get_action(self, state: TableState, actions: List[Action]) -> Action:
        #  С вероятностью epsilon совершаем случайное действие
        if np.random.rand() < self.epsilon:
            return np.random.choice(actions)
        # Иначе оптимальное с точки зрения Q-функции
        return self._get_best_action(state, actions)

    def __repr__(self):
        if self.epsilon == 0:
            return 'GreedyStrategy'
        else:
            return f'EpsilonGreedyStrategy(epsilon={round(self.epsilon, 3)})'
