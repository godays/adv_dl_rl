from __future__ import annotations

import random
from dataclasses import dataclass, fields, is_dataclass
from typing import Tuple, List, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from contracts import Action, State, Strategy, TicTacToeWrapper
from qlearning import Afterstate
from tic_tac_toe import TicTacToe


def to_tuple(dclass) -> tuple:
    """Нерекурсивный вариант dataclasses.astuple"""
    if not is_dataclass(dclass):
        raise TypeError
    return tuple(getattr(dclass, field.name) for field in fields(dclass))


@dataclass(frozen=True)
class DQNAction(Action):
    """Возможный ход игрока"""
    i: int
    j: int
    index_1d: int

    @staticmethod
    def from_i_j_cols(i: int, j: int, n_cols: int) -> DQNAction:
        index_1d = i * n_cols + j
        return DQNAction(i, j, index_1d)

    @staticmethod
    def from_index_1d(index_1d: int, n_cols: int) -> DQNAction:
        i = index_1d // n_cols
        j = index_1d % n_cols
        return DQNAction(i, j, index_1d)


def empty_positions_to_actions(empty_positions: np.array, n_cols: int) -> List[DQNAction]:
    return [DQNAction.from_i_j_cols(*pos, n_cols) for pos in empty_positions]


@dataclass(frozen=True)
class DQNState(State):
    current_turn: int  # Чей сейчас ход
    board_hash: str  # Строковое представление состояния доски. Нужно для игры против табличных методов
    board: np.ndarray  # Состояние доски

    # def to_tensor(self) -> torch.Tensor:
    #     crosses_tensor = torch.tensor(self.board == 1, dtype=torch.float32).unsqueeze(0)
    #     naughts_tensor = torch.tensor(self.board == -1, dtype=torch.float32).unsqueeze(0)
    #     turn_tensor = torch.empty_like(crosses_tensor).fill_(self.current_turn)
    #     board_tensor = torch.cat([turn_tensor, crosses_tensor, naughts_tensor]).unsqueeze(0)
    #     return board_tensor

    def to_tensor(self) -> torch.Tensor:
        crosses_tensor = torch.tensor(self.board == 1, dtype=torch.float32).unsqueeze(0)
        naughts_tensor = torch.tensor(self.board == -1, dtype=torch.float32).unsqueeze(0)
        if self.current_turn == 1:
            board_tensor = torch.cat([crosses_tensor, naughts_tensor]).unsqueeze(0)
        else:
            board_tensor = torch.cat([naughts_tensor, crosses_tensor]).unsqueeze(0)
        return board_tensor

    def get_afterstate(self, action: DQNAction) -> Afterstate:
        """Нужно для совместимости с табличными методами"""
        afterstate_hash = (self.board_hash[:action.index_1d]
                           + str(self.current_turn + 1)
                           + self.board_hash[action.index_1d + 1:])
        return Afterstate(afterstate_hash)


class TicTacToeDQNWrapper(TicTacToeWrapper):
    """Оборачивает оригинальное окружение, чтобы возвращать состояния и действия в виде объектов DQNState и DQNAction,
    пригодных для обучения DQN"""

    def __init__(self, env: TicTacToe):
        super().__init__(env)

    def reset(self) -> Tuple[DQNState, List[DQNAction]]:
        self.env.reset()
        n_cols = self.env.n_cols
        env_state, reward, done, _ = self.env.reset()
        board_hash, empty_positions, current_turn = env_state
        board = self.env.board.copy()
        actions = empty_positions_to_actions(empty_positions, n_cols)
        state = DQNState(current_turn, board_hash, board)
        return state, actions

    def step(self, action: DQNAction) -> Tuple[DQNState, List[DQNAction], int, bool]:
        n_cols = self.env.n_cols
        env_state, crosses_reward, done, _ = self.env.step((action.i, action.j))
        board_hash, empty_positions, current_turn = env_state
        board = self.env.board.copy()
        actions = empty_positions_to_actions(empty_positions, n_cols)
        state = DQNState(current_turn, board_hash, board)
        #  crosses_reward - реворд для крестиков
        #  реворд для ноликов == -1*crosses_reward
        #  исключение, если reward=-10: тогда это reward последнего сходившего игрока
        return state, actions, crosses_reward, done


class DQNStrategy(Strategy):
    def __init__(self, model: torch.nn.Module, epsilon: float, n_cols: int, device: str = 'cpu'):
        self.model = model
        self.epsilon = epsilon
        self.n_cols = n_cols
        self.device = device

    def _get_best_action(self, state: DQNState) -> DQNAction:
        """Возвращает действие, максимизирующее Q(state, a)"""
        self.model.eval()
        input_tensor = state.to_tensor().to(self.device)
        with torch.no_grad():
            q_values = self.model(input_tensor)
        best_action_1d = q_values.view(-1).max(0).indices.item()
        return DQNAction.from_index_1d(best_action_1d, self.n_cols)

    def get_action(self, state: DQNState, actions: List[DQNAction]) -> DQNAction:
        #  С вероятностью epsilon совершаем случайное действие
        if np.random.rand() < self.epsilon:
            return np.random.choice(actions)
        # Иначе оптимальное с точки зрения Q-функции
        return self._get_best_action(state)

    def __repr__(self):
        if self.epsilon == 0:
            return 'GreedyDQNStrategy'
        else:
            return f'EpsilonGreedyDQNStrategy(epsilon={round(self.epsilon, 3)})'


@dataclass
class Experience:
    state: DQNState
    action: DQNAction
    next_state: DQNState
    reward: int
    done: bool


@dataclass
class ExperienceBatch:
    state_batch: torch.Tensor
    action_batch: torch.Tensor
    next_state_batch: torch.Tensor
    reward_batch: torch.Tensor
    done_batch: torch.Tensor


class ReplayMemory:
    def __init__(self, capacity: int, device: str = 'cpu'):
        self.capacity: int = capacity
        self.memory: List[Experience] = []
        self.position: int = 0
        self.device = device

    def store(self, experience: Experience) -> None:
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def extend(self, experiences: Iterable[Experience]) -> None:
        for e in experiences:
            self.store(e)

    def sample(self, batch_size: int) -> ExperienceBatch:
        samples = random.sample(self.memory, batch_size)
        states, actions, next_states, rewards, done = zip(*[to_tuple(s) for s in samples])
        state_batch = torch.cat([s.to_tensor() for s in states]).to(self.device)
        action_ids = [a.index_1d for a in actions]
        action_batch = torch.tensor(action_ids, dtype=torch.int64).unsqueeze(-1).to(self.device)
        next_state_batch = torch.cat([s.to_tensor() for s in next_states]).to(self.device)
        reward_batch = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(done, dtype=torch.bool).to(self.device)
        return ExperienceBatch(state_batch, action_batch, next_state_batch, reward_batch, done_batch)

    def __len__(self) -> int:
        return len(self.memory)


class DQNTrainer:
    def __init__(self,
                 env: TicTacToeDQNWrapper,
                 strategy: DQNStrategy,
                 batch_size: int = 64,
                 learning_rate: float = 0.001,
                 replay_memory_size: int = 1000,
                 device: str = 'cpu'):
        self.env = env
        self.strategy = strategy
        self.memory = ReplayMemory(replay_memory_size, device)
        self.optimizer = torch.optim.Adam(strategy.model.parameters(), learning_rate)
        self.batch_size = batch_size
        self.device = device

    def _generate_experiences(self,
                              state_actions: List[Tuple[DQNState, DQNAction]],
                              final_state: DQNState,
                              reward: int) -> List[Experience]:
        experiences: List[Experience] = []
        reversed_state_actions = list(reversed(state_actions))
        current_state, current_action = reversed_state_actions[0]
        next_state = final_state
        #  Для последнего хода запоминаем, что игра после него закончилась.
        #  Оценку для него будем подтягивать сразу к реворду
        experiences.append(Experience(current_state, current_action, next_state, reward, done=True))
        next_state = current_state
        for current_state, current_action in reversed_state_actions[1:]:
            #  Можем указать reward=0, потому что при флаге done=False всё равно будем брать
            #  только оценку следующего лучшего хода
            experiences.append(Experience(current_state, current_action, next_state, reward=0, done=False))
            next_state = current_state
        return experiences

    def run_episode(self):
        """Генерируем эпизод, играя против себя, и сохраняем ход игры в память"""
        crosses_state_actions = []
        naughts_state_actions = []
        state, actions = self.env.reset()
        while True:
            action = self.strategy.get_action(state, actions)
            if self.env.env.curTurn == 1:
                crosses_state_actions.append((state, action))
            else:
                naughts_state_actions.append((state, action))
            state, actions, crosses_reward, done = self.env.step(action)
            if done:
                #  Начинаем генерировать единицы опыта
                if crosses_reward == -10:
                    # Замечание: независимо от того, кто именно совершил нелегальный ход, это число равно -10
                    # Нет смысла поощрять стратегию, если оппонент совершил нелегальный ход
                    if self.env.env.curTurn == 1:
                        crosses_experiences = self._generate_experiences(crosses_state_actions, state, -10)
                        self.memory.extend(crosses_experiences)
                    else:
                        naughts_experiences = self._generate_experiences(naughts_state_actions, state, -10)
                        self.memory.extend(naughts_experiences)
                    return
                crosses_experiences = self._generate_experiences(crosses_state_actions, state, crosses_reward)
                self.memory.extend(crosses_experiences)
                naughts_reward = -crosses_reward
                naughts_experiences = self._generate_experiences(naughts_state_actions, state, naughts_reward)
                self.memory.extend(naughts_experiences)
                return

    def train_on_batch(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)

        self.strategy.model.train()
        # Значения Q-функции для всех действий
        predicted_q = self.strategy.model(batch.state_batch).view(self.batch_size, -1)
        # Значения Q-функции для совершённых действий
        predicted_q_a = predicted_q.gather(1, batch.action_batch).reshape([self.batch_size])

        # оцениваем ожидаемые значения после этого действия
        max_next_q = self.strategy.model(batch.next_state_batch).view(self.batch_size, -1).detach().max(1)[0]

        # Для последнего действия берём сразу реворд, для остальных - оценку следующего лучшего действия
        q_target = torch.where(batch.done_batch, batch.reward_batch, max_next_q)

        # и хотим, чтобы predicted_q_a было похоже на q_target -- это и есть суть Q-обучения
        loss = F.smooth_l1_loss(predicted_q_a, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.strategy.model.eval()

    def fit(self, iterations: int, fit_each_step: int = 2):
        for i in trange(iterations, position=0, desc='Fit'):
            self.run_episode()
            if i % fit_each_step == 0:
                self.train_on_batch()
