import gym
import numpy as np
from tqdm import tqdm
from itertools import product

A = (0, 1)
S = list(product(range(4, 32), range(1, 11), (True, False)))
S = [s for s in S if not s[2] or (s[2] and 12 <= s[0] <= 21)]
s2idx = {state: i for i, state in enumerate(S)}
pi_dummy = [int(state[0] not in {19, 20, 21}) for state, idx in s2idx.items()]


def play_game(env, pi):
    observation = env.reset()
    done = False
    G = 0
    while not done:
        action = pi[s2idx[observation]]
        observation, reward, done, _ = env.step(action)
        G += reward
    return G

def evaluate_pi(env, pi=pi_dummy, n=100_000):
    rewards = []
    for _ in (range(n)):
        reward = play_game(env, pi)
        rewards.append(reward)
    return np.mean(rewards)