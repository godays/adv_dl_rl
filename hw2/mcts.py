from __future__ import annotations

from copy import deepcopy
from typing import Optional, List, Dict

import numpy as np
from joblib import Parallel, delayed, parallel_backend

from contracts import Action, State, Strategy, RandomStrategy, TicTacToeWrapper
from dqn import DQNAction, DQNState, TicTacToeDQNWrapper


def rollout(env: TicTacToeWrapper,
            action: Action,
            strategy: Optional[Strategy] = None) -> int:
    """Доигрывает партию по стратегии strategy, совершая вначале действие action"""
    env = deepcopy(env)
    strategy = strategy or RandomStrategy()
    current_player = env.env.curTurn
    state, actions, crosses_reward, done = env.step(action)
    while not done:
        action = strategy.get_action(state, actions)
        state, actions, crosses_reward, done = env.step(action)
    if crosses_reward == -10:
        raise NotImplementedError
    return crosses_reward * current_player


class RolloutStrategy(Strategy):
    def __init__(self,
                 env: TicTacToeWrapper,
                 n_iter: int,
                 n_jobs: int = 1,
                 base_strategy: Optional[Strategy] = None):
        assert n_jobs > 0
        self.env = env
        self.n_iter = n_iter
        self.base_strategy = base_strategy or RandomStrategy()
        self.n_jobs = n_jobs

    def eval_action(self, action: Action) -> float:
        return np.mean([rollout(self.env, action, self.base_strategy)
                        for _ in range(self.n_iter)])

    def get_action(self, state: State, actions: List[Action]) -> Action:
        if self.n_jobs == 1 or len(actions) < 2:
            action_scores = [self.eval_action(action) for action in actions]
        else:
            with parallel_backend('loky', n_jobs=self.n_jobs):
                tasks = (delayed(self.eval_action)(action) for action in actions)
                action_scores = Parallel()(tasks)
        return actions[np.argmax(action_scores)]


class MCTSNode:
    def __init__(self,
                 current_turn: int,
                 board_hash: str,
                 parent: Optional[MCTSNode],
                 action: Optional[DQNAction] = None):
        self.current_turn = current_turn  # Игрок, принимающий решение в данном состоянии
        self.board_hash = board_hash
        if parent is not None:
            assert self.current_turn == parent.current_turn * -1
        self.parent = parent
        self.action = action  # Ход, который привёл к текущему состоянию
        self.children: Optional[List[MCTSNode]] = None
        self.total_visits: int = 0
        self.wins: int = 0
        self.ties: int = 0

    def get_score(self) -> float:
        if self.total_visits == 0:
            return 0
        return (self.wins + 0.5 * self.ties) / self.total_visits

    def is_leaf(self) -> bool:
        return self.children is None or any(c.total_visits == 0 for c in self.children)


class MCTSStrategy(Strategy):

    def __init__(self,
                 env: TicTacToeDQNWrapper,
                 base_strategy: Optional[Strategy] = None,
                 num_selections: int = 100,
                 num_rollouts: int = 100,
                 C: float = np.sqrt(2)):
        self.env = env
        self.base_strategy = base_strategy or RandomStrategy()
        self.num_selections = num_selections
        self.num_rollouts = num_rollouts
        self.C = C
        self.hashes_to_nodes: Dict[str, MCTSNode] = dict()

    def make_node(self,
                  state: DQNState,
                  action: Optional[Action] = None,
                  parent: Optional[MCTSNode] = None) -> MCTSNode:
        node = MCTSNode(state.current_turn, state.board_hash, parent, action)
        self.hashes_to_nodes[state.board_hash] = node
        return node

    def get_node(self, state: DQNState) -> MCTSNode:
        if state.board_hash in self.hashes_to_nodes:
            return self.hashes_to_nodes[state.board_hash]
        return self.make_node(state)

    def make_child(self, node: MCTSNode, action: DQNAction) -> MCTSNode:
        board_hash = node.board_hash
        action_index = action.index_1d
        current_turn = node.current_turn
        next_state_hash = (board_hash[:action_index]
                           + str(current_turn + 1)
                           + board_hash[action_index + 1:])
        child = MCTSNode(current_turn * -1, next_state_hash, node, action)
        self.hashes_to_nodes[next_state_hash] = child
        return child

    def add_children(self, node: MCTSNode, actions: List[DQNAction]) -> None:
        if node.children is None:
            node.children = []
            for action in actions:
                node.children.append(self.make_child(node, action))

    def get_uct(self, node: MCTSNode) -> float:
        assert node.parent is not None
        return node.get_score() + self.C * np.sqrt(np.log(node.parent.total_visits + 1) / (node.total_visits + 1))

    def select_child(self, node: MCTSNode) -> MCTSNode:
        assert node.children is not None
        key = lambda n: self.get_uct(n)
        return max(node.children, key=key)

    def backpropagate(self, node: MCTSNode, wins: int, ties: int, total: int) -> None:
        losses = total - (wins + ties)
        while node is not None:
            node.total_visits += total
            node.wins += wins
            node.ties += ties
            node = node.parent
            wins, losses = losses, wins

    def get_action(self, state: DQNState, actions: List[DQNAction]) -> DQNAction:
        root = self.get_node(state)
        self.add_children(root, actions)
        root_actions = actions
        for _ in range(self.num_selections):
            # Selection
            node = root
            node_actions = root_actions
            env = deepcopy(self.env)
            done = False
            while not done and not node.is_leaf():
                node = self.select_child(node)
                state, node_actions, crosses_reward, done = env.step(node.action)
            if done:
                player = -env.env.curTurn
                win, tie = 0, 0
                if crosses_reward == 0:
                    tie = 1
                else:
                    reward = crosses_reward * player
                    win = reward == 1
                self.backpropagate(node, win * self.num_rollouts, tie * self.num_rollouts, self.num_rollouts)
                continue
            # Expansion
            self.add_children(node, node_actions)
            next_node = self.select_child(node)
            # Simulation
            # Начинаем игру из состояния state, ему соответствует узел node
            # Первым ходом совершаем next_node.action и попадаем в next_node
            # Rewards соответстуют игроку из состояния node
            # next_node - состояние его противника
            rewards = np.array([rollout(env, next_node.action, self.base_strategy)
                                for _ in range(self.num_rollouts)])
            next_node_wins = rewards[rewards == -1].sum()
            ties = rewards[rewards == 0].sum()
            # Backpropagation
            self.backpropagate(next_node, next_node_wins, ties, self.num_rollouts)
        best_node = max(root.children, key=lambda node: node.get_score())
        return best_node.action
