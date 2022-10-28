import gym
import numpy as np
from tqdm import tqdm
from itertools import product
import random
import matplotlib.pyplot as plt
from func_19_20_21 import play_game, evaluate_pi

A = (0, 1)
S = list(product(range(4, 32), range(1, 11), (True, False)))
S = [s for s in S if not s[2] or (s[2] and 12 <= s[0] <= 21)]
s2idx = {state: i for i, state in enumerate(S)}



def get_pi_by_Q(Q):
    return np.argmax(Q, axis=1)

def episode(env, Q, A, alpha=0.05, epsilon=0.1, gamma=1):
    s = s2idx[env.reset()]
    done = False
    get_actions_from_env = hasattr(env, 'get_actions')
        
    while not done:
        pi = get_pi_by_Q(Q)
        a = pi[s] if random.random() < (1 - epsilon) else random.choice(env.get_actions() if get_actions_from_env else A)
        observation, r_new, done, _ = env.step(a)
        s_new = s2idx[observation]
        Q[s, a] = Q[s, a] + alpha * (r_new + gamma * max(Q[s_new]) - Q[s, a])
        s = s_new
    return Q

def q_learning(env, A, Q, episodes=100_000, alpha=0.009, epsilon=0.85, gamma=1):
    for _ in (range(episodes)):
        Q = episode(env=env, Q=Q, A=A, alpha=alpha, epsilon=epsilon, gamma=gamma)
        
    return Q
        

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def plot_q_learinig(env, A, alpha, epsilon, episodes=1000, n_experiments=50, n_eval=100):

    Q_arr = [np.zeros((len(S), len(A)))] * n_experiments
    pi_arr = [get_pi_by_Q(Q) for Q in Q_arr]
    results = []
    episode_results = []
    episode_nums = []

    for eps in range(episodes):
        experiments_result = []
        for i in range(n_experiments):
            Q_arr[i] = episode(env, Q_arr[i], A, alpha=alpha, epsilon=epsilon)
            pi_arr[i] = get_pi_by_Q(Q_arr[i])
            experiments_result.append(evaluate_pi(env, pi_arr[i], n=n_eval))

        episode_results.append(np.mean(experiments_result))
        episode_nums.append(eps)

    plt.figure(figsize=(16, 8))
    plt.plot(episode_nums, episode_results, linewidth=1)
    plt.xlabel('Число эпизодов обучения')
    plt.ylabel('Награда')
    print(f'Награда каждого эпизода усреднена по {n_experiments} экспериментам и посчитана на {n_eval} раздачах. Всего {episodes} эпизодов')
    plt.show()