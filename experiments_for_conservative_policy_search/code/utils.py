from itertools import product
from Categories import STATE_OCCUPANCY
import environment
import numpy as np


# Checked 16.5.2022
def get_optimal_policy(env, pi_1, pi_2, indices, n_iter):
    q_values = get_optimal_q(env, pi_1, pi_2, indices, n_iter)
    pi_2_opt = np.zeros(pi_2.shape)
    pi_2_opt[range(pi_2_opt.shape[0]), np.argmax(q_values, axis=1)] = 1

    ans = np.zeros((env.n_states, env.n_actions))
    ans[indices] = pi_2[indices]
    ans[~indices] = pi_2_opt[~indices]
    return ans


# CHECKED 16.5.2022
def get_optimal_q(env: environment, pi_1, pi_2, indices, n_iter):
    q_values = np.zeros((env.n_states, env.n_actions))
    reduced_R = np.zeros((env.n_states, env.n_actions))
    reduced_P = np.zeros((env.n_states, env.n_actions, env.n_states))
    for s in range(env.n_states):
        reduced_R[s, :] = np.einsum('i,ij', pi_1[s, :], env.R[s, :, :])
        reduced_P[s, :, :] = np.einsum('i,ijk', pi_1[s, :], env.P[s, :, :, :])
    for _ in range(n_iter):
        V = np.zeros(env.n_states)
        V[indices] = np.multiply(q_values[indices], pi_2[indices]).sum(axis=1)
        V[~indices] = np.max(q_values[~indices], axis=1)
        q_values = np.copy(reduced_R)
        q_values += env.gamma * np.einsum('ijk,k', reduced_P, V)
    return q_values


def get_stationary_dist(env: environment, pi_1, pi_2, n_iter, state_occupancy):
    mu = np.ones(env.n_states) / env.n_states
    reduced_P = np.zeros((env.n_states, env.n_states))
    for s in range(env.n_states):
        reduced_P[s, :] = np.einsum('i,ij', pi_2[s, :], np.einsum('i,ijk', pi_1[s, :], env.P[s, :, :, :]))
    if state_occupancy == STATE_OCCUPANCY.AVERAGE:
        for _ in range(n_iter):
            mu = reduced_P.T @ mu
    else:
        for _ in range(n_iter):
            mu = ((1 - env.gamma) * env.initial_dist + env.gamma * (reduced_P.T @ mu))
    return mu


# Take pi_2, output all its neighbors that differ only in s, a
# Checked 16.5.2022
def get_neighbors(pi_2):
    ans = []
    for s in range(pi_2.shape[0]):
        for a in range(pi_2.shape[1]):
            if pi_2[s, a] == 1: continue
            c = np.copy(pi_2)
            c[s] = np.zeros(pi_2.shape[1])
            c[s, a] = 1
            ans.append(c)
    return ans


# Checked 16.5.2022
def get_expected_reward(env: environment, pi_2):
    res = np.zeros((env.n_states, env.n_actions))
    for s in range(env.n_states):
        res[s, :] = env.R[s, :, :] @ pi_2[s, :]
    return res
