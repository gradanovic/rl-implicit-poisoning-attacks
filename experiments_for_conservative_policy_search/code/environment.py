## This file consists of the definitions of different environments
## If you want to create Environment, please follow the same structures as the implemented functions i.e.
## create_inventory_control


import numpy as np

class Environment:
    def __init__(self, n_states, n_actions, reward, transition, gamma, initial_dist):
        self.n_states = n_states
        self.n_actions = n_actions
        self.P = transition
        self.P_ref = np.copy(transition)
        self.R = reward
        self.R_ref = np.copy(reward)
        self.gamma = gamma
        self.initial_dist = initial_dist

    def is_valid_policy(self, pi_2, mu):
        for s in range(self.n_states):
            for a_2 in range(self.n_actions):
                if pi_2[s, a_2] > 0.01 and self.R[s, 0, a_2] < -1e7 and mu[s] > 0.001:
                    return False
        return True

    # Given an initial policy for agent 1, this function sets the value of influence of agent 1 in the environment
    def change_influence(self, influence, pi_1_ref):
        self.R = np.zeros(self.R_ref.shape)
        self.P = np.zeros(self.P_ref.shape)

        for s in range(self.R_ref.shape[0]):
            self.R[s, :, :] = influence * self.R_ref[s, :, :] + (1 - influence) * pi_1_ref[s, :] @ self.R_ref[s, :, :]
        for s in range(self.P_ref.shape[0]):
            self.P[s, :, :, :] = influence * self.P_ref[s, :, :, :] + \
                                 (1 - influence) * np.einsum('i,ijk', pi_1_ref[s, :], self.P_ref[s, :, :, :])


# Old Simple Chain and Old Navigation Environments
##############################################
# add a path from s1 to s2 with probability of P1 if both agents go in direction dir,
# with probability of 1-P1 choose the destination randomly
# add a path from s1 to s2 with probability of P2 if agent 2 goes towards dir, and agent 1 goes the opposite direction,
# with probability of 1-P2, choose the destination randomly
def add_path(s1, dir, s2, P1, P2, P, n_states):
    P[s1, dir, dir, s2] = P1
    P[s1, dir, dir, :] += (1 - P1) / n_states

    P[s1, 1 - dir, dir, s2] = P2
    P[s1, 1 - dir, dir, :] += (1 - P2) / n_states


def create_old_chain_env(alpha, beta, P1, P2, gamma):
    # set number of states and actions space for both agents
    n_states = 4
    n_actions = 2

    # define the reward function
    R = np.zeros((n_states, n_actions, n_actions))
    for s in range(n_states):
        R[s, :, :] = np.array([[alpha, beta],
                               [beta, alpha]])

    # define the transition function
    P = np.zeros((n_states, n_actions, n_actions, n_states))
    add_path(0, 0, 0, P1, P2, P, n_states)
    add_path(0, 1, 1, P1, P2, P, n_states)
    add_path(1, 0, 0, P1, P2, P, n_states)
    add_path(1, 1, 2, P1, P2, P, n_states)
    add_path(2, 0, 1, P1, P2, P, n_states)
    add_path(2, 1, 3, P1, P2, P, n_states)
    add_path(3, 0, 2, P1, P2, P, n_states)
    add_path(3, 1, 3, P1, P2, P, n_states)

    # set the initial states
    initial_dist = np.zeros(n_states)
    initial_dist[0] = 1

    return Environment(n_states, n_actions, R, P, gamma, initial_dist=initial_dist)


def create_old_navigation_chain(alpha, beta, P1, P2, gamma):
    # set number of states and actions space for both agents
    n_states = 9
    n_actions = 2

    # define the reward function
    R = np.zeros((n_states, n_actions, n_actions))
    for s in range(n_states):
        R[s, :, :] = np.array([[alpha, beta],
                               [beta, alpha]])

    # define the transition function
    P = np.zeros((n_states, n_actions, n_actions, n_states))
    add_path(0, 0, 0, P1, P2, P, n_states)
    add_path(0, 1, 1, P1, P2, P, n_states)
    add_path(1, 0, 0, P1, P2, P, n_states)
    add_path(1, 1, 2, P1, P2, P, n_states)
    add_path(2, 0, 1, P1, P2, P, n_states)
    add_path(2, 1, 3, P1, P2, P, n_states)
    add_path(3, 0, 2, P1, P2, P, n_states)
    add_path(3, 1, 4, P1, P2, P, n_states)
    add_path(4, 0, 5, P1, P2, P, n_states)
    add_path(4, 1, 7, P1, P2, P, n_states)
    add_path(5, 0, 6, P1, P2, P, n_states)
    add_path(5, 1, 4, P1, P2, P, n_states)
    add_path(6, 0, 6, P1, P2, P, n_states)
    add_path(6, 1, 5, P1, P2, P, n_states)
    add_path(7, 0, 4, P1, P2, P, n_states)
    add_path(7, 1, 8, P1, P2, P, n_states)
    add_path(8, 0, 7, P1, P2, P, n_states)
    add_path(8, 1, 8, P1, P2, P, n_states)

    # set the initial states
    initial_dist = np.zeros(n_states)
    initial_dist[0] = 1

    return Environment(n_states, n_actions, reward=R, transition=P, gamma=gamma, initial_dist=initial_dist)


##############################################

# Simple Chain and Environment
##############################################
# add a path from s_1 to s_2 with probability of p, if agents take a_1 and a_2 respectively,
# otherwise choose the destination randomly.
def add_simple_path(s_1, a_1, a_2, s_2, P, p, n_states):
    P[s_1, a_1, a_2, s_2] += p
    P[s_1, a_1, a_2, :] += (1 - p) / n_states


# if agents take a_1 and a_2 respectively in state s_1, choose the destination randomly.
def add_random_path(s_1, a_1, a_2, P, n_states):
    P[s_1, a_1, a_2, :] = 1 / n_states


def create_chain(p, gamma):
    # set number of states and actions space for both agents
    n_states = 5
    n_actions = 2

    # define the reward function
    R = np.zeros((n_states, n_actions, n_actions))
    for s in range(n_states):
        R[s, :, :] = np.array([[0, 0],
                               [0, 5]])
    R[0] += 5
    R[2] += 50
    R[4] += 5

    # define the transition function
    P = np.zeros((n_states, n_actions, n_actions, n_states))
    # s_0
    add_simple_path(0, 0, 0, 0, P, p, n_states)
    add_random_path(0, 0, 1, P, n_states)
    add_random_path(0, 1, 0, P, n_states)
    add_simple_path(0, 1, 1, 1, P, p, n_states)
    # s_1
    add_simple_path(1, 0, 0, 0, P, p, n_states)
    add_simple_path(1, 0, 1, 0, P, p, n_states)
    add_simple_path(1, 1, 0, 2, P, p, n_states)
    add_simple_path(1, 1, 1, 2, P, p, n_states)
    # s_2
    add_simple_path(2, 0, 0, 3, P, p, n_states)
    add_simple_path(2, 0, 1, 3, P, p, n_states)
    add_simple_path(2, 1, 0, 3, P, p, n_states)
    add_simple_path(2, 1, 1, 3, P, p, n_states)
    # s_3
    add_simple_path(3, 0, 0, 1, P, p, n_states)
    add_simple_path(3, 0, 1, 4, P, p, n_states)
    add_simple_path(3, 1, 0, 1, P, p, n_states)
    add_simple_path(3, 1, 1, 4, P, p, n_states)
    # s_4
    add_simple_path(4, 0, 0, 3, P, p, n_states)
    add_random_path(4, 0, 1, P, n_states)
    add_random_path(4, 1, 0, P, n_states)
    add_simple_path(4, 1, 1, 4, P, p, n_states)

    # set the initial states
    initial_dist = np.zeros(n_states)
    initial_dist[0] = 1

    return Environment(n_states, n_actions, R, P, gamma, initial_dist=initial_dist)


def create_navigation(p, gamma):
    # set number of states and actions space for both agents
    n_states = 9
    n_actions = 2

    # define the reward function
    R = np.zeros((n_states, n_actions, n_actions))
    for s in range(n_states):
        R[s, :, :] = np.array([[5, -5],
                               [-5, 5]])
    R[0] += 5
    R[2] += 50
    R[4] += 5
    # R[6] += 5
    R[8] += 5

    # define the transition function
    P = np.zeros((n_states, n_actions, n_actions, n_states))
    # s_0
    add_simple_path(0, 0, 0, 0, P, p, n_states)
    add_random_path(0, 0, 1, P, n_states)
    add_random_path(0, 1, 0, P, n_states)
    add_simple_path(0, 1, 1, 1, P, p, n_states)
    # s_1
    add_simple_path(1, 0, 0, 0, P, p, n_states)
    add_simple_path(1, 0, 1, 0, P, p, n_states)
    add_simple_path(1, 1, 0, 2, P, p, n_states)
    add_simple_path(1, 1, 1, 2, P, p, n_states)
    # s_2
    add_simple_path(2, 0, 0, 3, P, p, n_states)
    add_simple_path(2, 0, 1, 3, P, p, n_states)
    add_simple_path(2, 1, 0, 3, P, p, n_states)
    add_simple_path(2, 1, 1, 3, P, p, n_states)
    # s_3
    add_simple_path(3, 0, 0, 1, P, p, n_states)
    add_simple_path(3, 0, 1, 4, P, p, n_states)
    add_simple_path(3, 1, 0, 1, P, p, n_states)
    add_simple_path(3, 1, 1, 4, P, p, n_states)
    # s_4
    add_simple_path(4, 0, 0, 5, P, p, n_states)
    add_random_path(4, 0, 1, P, n_states)
    add_random_path(4, 1, 0, P, n_states)
    add_simple_path(4, 1, 1, 7, P, p, n_states)
    # s_5
    add_simple_path(5, 0, 0, 6, P, p, n_states)
    add_random_path(5, 0, 1, P, n_states)
    add_random_path(5, 1, 0, P, n_states)
    add_simple_path(5, 1, 1, 4, P, p, n_states)
    # s_6
    add_simple_path(6, 0, 0, 6, P, p, n_states)
    add_random_path(6, 0, 1, P, n_states)
    add_random_path(6, 1, 0, P, n_states)
    add_simple_path(6, 1, 1, 5, P, p, n_states)
    # s_7
    add_simple_path(7, 0, 0, 4, P, p, n_states)
    add_random_path(7, 0, 1, P, n_states)
    add_random_path(7, 1, 0, P, n_states)
    add_simple_path(7, 1, 1, 8, P, p, n_states)
    # s_8
    add_simple_path(8, 0, 0, 7, P, p, n_states)
    add_random_path(8, 0, 1, P, n_states)
    add_random_path(8, 1, 0, P, n_states)
    add_simple_path(8, 1, 1, 8, P, p, n_states)

    # set the initial states
    initial_dist = np.zeros(n_states)
    initial_dist[0] = 1

    return Environment(n_states, n_actions, R, P, gamma, initial_dist=initial_dist)


##############################################


# Inventory Example
##############################################
# Note that the infeasible actions have a reward of -10^10
def create_inventory_control(M, gamma, reject=False):

    n_states = M
    n_actions = M

    # define the cost of buying items
    buy = np.zeros(M)
    for i in range(1, M):
        buy[i] = 4 + 2 * i

    # define the cost of maintaining items
    hold = np.zeros(M)
    for i in range(M):
        hold[i] = 0 + 1 * i

    # define the value of selling items
    sell = np.zeros(M)
    for i in range(M):
        sell[i] = 10 * i

    P = np.zeros((n_states, n_actions, n_actions, n_states))
    R = np.zeros((n_states, n_actions, n_actions))

    # define the reward and transition function
    for s_1 in range(n_states):
        for a_1 in range(n_actions):
            for a_2 in range(n_actions):
                if s_1 + a_2 >= n_states:
                    R[s_1, a_1, a_2] = -1e10
                    P[s_1, a_1, a_2, s_1] = 1
                    continue
                s_2 = None
                if not reject:
                    s_2 = max(s_1 + a_2 - a_1, 0)
                else:
                    if s_1 + a_2 >= a_1:
                        s_2 = s_1 + a_2 - a_1
                    else:
                        s_2 = s_1 + a_2
                P[s_1, a_1, a_2, s_2] = 1
                R[s_1, a_1, a_2] = -(buy[a_2] + hold[s_1 + a_2] - sell[s_1 + a_2 - s_2])

    # set the initial states
    initial_dist = np.zeros(n_states)
    initial_dist[0] = 1

    return Environment(n_states, n_actions, reward=R, transition=P, gamma=gamma, initial_dist=initial_dist)
##############################################
