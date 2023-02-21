from environment import Environment
import utils
import numpy as np
import cvxpy as cp
import math
from tqdm import tqdm
from Categories import ALGORITHM, OPTIMIZER, STATE_OCCUPANCY, OPTIMIZER
import time
import matplotlib.pyplot as plt


# modify pi_2 to obtain pi_2_tilde
def get_improved_policy(env, pi_1, pi_2, indices, util_iter):
    return utils.get_optimal_policy(env, pi_1, pi_2, indices=indices, n_iter=util_iter)

def initialize_algorithm(env: Environment, pi_1_ref, pi_2_target, ergodicity, state_occupancy, util_iter, mu_th):
    indices = np.array([True] * env.n_states)
    if ergodicity:
        # If the env is ergodic, then it suffices to find neighbors in the first place, and continue with the found neighbors
        pi_2 = pi_2_target
        target_R = utils.get_expected_reward(env, pi_2)
        neighbors = utils.get_neighbors(pi_2)
        neighbors_R = []
        for neighbor in neighbors:
            neighbors_R.append(utils.get_expected_reward(env, neighbor))
    else:
        # if the env is non-ergodic, we need to find pi_2 (which is pi_2 tilde) instead of pi_2_target.
        # Moreover, because pi_1 is fully stochastic, the set of visited states when agent 2 executes pi_2_target
        # remains the same over iterations.
        target_mu = utils.get_stationary_dist(env, pi_1_ref, pi_2_target, n_iter=util_iter,
                                              state_occupancy=state_occupancy)
        indices = target_mu > mu_th
        pi_2 = utils.get_optimal_policy(env, pi_1_ref, pi_2_target, indices=indices, n_iter=util_iter)
        target_R = utils.get_expected_reward(env, pi_2)
        neighbors = utils.get_neighbors(pi_2)
        filtered_neighbors = []
        for neighbor in neighbors:
            neighbor_mu = utils.get_stationary_dist(env, pi_1_ref, neighbor, n_iter=util_iter,
                                                    state_occupancy=state_occupancy)
            if not env.is_valid_policy(neighbor, neighbor_mu):
                continue
            if np.sum(np.sum(np.abs(neighbor - pi_2), axis=1) * indices) < 0.01:
                continue
            filtered_neighbors.append(get_improved_policy(env, pi_1_ref, neighbor, indices, util_iter))
        neighbors = filtered_neighbors
        neighbors_R = []
        for neighbor in neighbors:
            neighbors_R.append(utils.get_expected_reward(env, neighbor))

    return pi_2, target_R, neighbors, neighbors_R, indices


def create_constraints(env: Environment, pi_1, pi_1_ref, pi_2, target_R, neighbors, neighbors_R, eps, algorithm,
                       state_occupancy, util_iter, delta, pi_1_th):
    # pi_1 has to be a valid policy for agent 1
    constraints = [pi_1 >= pi_1_th, pi_1 <= 1, pi_1 @ np.ones(env.n_actions) == 1]

    # the target policy has to better than its neighbors
    target_mu = utils.get_stationary_dist(env, pi_1_ref, pi_2, n_iter=util_iter, state_occupancy=state_occupancy)
    for i in range(len(neighbors)):
        neighbor = neighbors[i]
        neighbor_R = neighbors_R[i]
        neighbor_mu = utils.get_stationary_dist(env, pi_1_ref, neighbor, n_iter=util_iter,
                                                state_occupancy=state_occupancy)
        constraints.append(cp.sum(cp.multiply(cp.sum(cp.multiply(target_R, pi_1), axis=1), target_mu)) >=
                           cp.sum(cp.multiply(cp.sum(cp.multiply(neighbor_R, pi_1), axis=1), neighbor_mu))
                           + eps)

    # add a constraint in case that the new policy must be close to the previous one
    if algorithm == ALGORITHM.COPS or algorithm == ALGORITHM.CPS:
        constraints.append(cp.abs(pi_1 - pi_1_ref) <= delta)

    return constraints


def create_objective(pi_1, pi_1_0, eps, epsilon, lam):
    # define the objective function
    objective = cp.Minimize(cp.sum(cp.abs(pi_1 - pi_1_0)) - lam * cp.minimum(eps, epsilon * 1.1))
    return objective


def get_current_eps(env: Environment, pi_1_ref, pi_2, target_R, neighbors, neighbors_R, state_occupancy, util_iter):
    # measure the current gap between pi_2_target and its neighbors
    target_mu = utils.get_stationary_dist(env, pi_1_ref, pi_2, n_iter=util_iter, state_occupancy=state_occupancy)
    current_eps = np.inf
    for i in range(len(neighbors)):
        neighbor = neighbors[i]
        neighbor_R = neighbors_R[i]
        neighbor_mu = utils.get_stationary_dist(env, pi_1_ref, neighbor, n_iter=util_iter,
                                                state_occupancy=state_occupancy)

        tmp = np.multiply(np.sum(np.multiply(target_R, pi_1_ref), axis=1), target_mu).sum() - \
              np.multiply(np.sum(np.multiply(neighbor_R, pi_1_ref), axis=1), neighbor_mu).sum()

        current_eps = min(current_eps, tmp)

    return current_eps


def renew_pi_2(env: Environment, pi_1_ref, pi_2_target, neighbors, indices, util_iter):
    # if the env is non-ergodic, we need to compute pi_2_tilde in the beginning of each iteration
    # and also compute the new neighbors
    pi_2 = utils.get_optimal_policy(env, pi_1_ref, pi_2_target, indices=indices, n_iter=util_iter)
    new_neighbors = []
    new_neighbors_R = []
    for neighbor in neighbors:
        neighbor = get_improved_policy(env, pi_1_ref, neighbor, indices, util_iter)
        new_neighbors.append(neighbor)
        new_neighbors_R.append(utils.get_expected_reward(env, neighbor))
    return pi_2, new_neighbors, new_neighbors_R


def heuristic_algorithm(env: Environment, pi_1_ref, pi_1_naive, pi_2_target, epsilon, delta, lam, algorithm, n_iter,
                        util_iter, ergodicity, state_occupancy=STATE_OCCUPANCY.WEIGHTED, optimizer=OPTIMIZER.NORMAL):
    if algorithm == ALGORITHM.COPS:
        lam = 100

    print(epsilon)

    # initialize variables
    pi_1_0 = np.copy(pi_1_ref)
    pi_1_ref = np.copy(pi_1_ref)
    mu_th = 0.0001
    pi_1_th = 0.001

    pi_2, target_R, neighbors, neighbors_R, indices = initialize_algorithm(env, pi_1_ref, pi_2_target,
                                                                           ergodicity=ergodicity,
                                                                           state_occupancy=state_occupancy,
                                                                           util_iter=util_iter, mu_th=mu_th)
    best_policy = pi_1_0
    best_eps = -10

    if algorithm == ALGORITHM.NAIVE:
        pi_2, target_R, neighbors, neighbors_R, indices = initialize_algorithm(env, pi_1_naive, pi_2_target,
                                                                               ergodicity=ergodicity,
                                                                               state_occupancy=state_occupancy,
                                                                               util_iter=util_iter, mu_th=mu_th)

        return get_current_eps(env, pi_1_naive, pi_2, target_R, neighbors, neighbors_R, state_occupancy,
                               util_iter), np.abs(pi_1_0 - pi_1_naive).sum(), []

    Eps = []
    times = []
    for _ in tqdm(range(n_iter)):
        start_time = time.time()

        if ergodicity == False:
            pi_2, neighbors, neighbors_R = renew_pi_2(env, pi_1_ref, pi_2_target, neighbors, indices, util_iter)

        current_eps = get_current_eps(env, pi_1_ref, pi_2, target_R, neighbors, neighbors_R, state_occupancy, util_iter)

        if current_eps > epsilon:
            if best_eps < epsilon or np.abs(pi_1_0 - pi_1_ref).sum() < np.abs(pi_1_0 - best_policy).sum():
                best_eps = current_eps
                best_policy = pi_1_ref

        # define variables
        pi_1 = cp.Variable((env.n_states, env.n_actions))
        eps = None
        if optimizer == OPTIMIZER.NORMAL:
            eps = cp.Variable()

        # define constraints
        constraints = create_constraints(env, pi_1, pi_1_ref, pi_2, target_R, neighbors, neighbors_R, eps, algorithm,
                                         state_occupancy, util_iter, delta, pi_1_th)

        objective = create_objective(pi_1, pi_1_0, eps, epsilon, lam)

        prob = cp.Problem(objective, constraints)
        result = prob.solve()

        if np.max(np.abs(pi_1.value - pi_1_ref)) < 1e-6:
            break

        pi_1_ref = np.copy(pi_1.value)
        Eps.append(current_eps)
        times.append(time.time() - start_time)


    # print(times)
    print(f'final result of this run: {current_eps}, {best_eps}, {np.abs(pi_1_0 - best_policy).sum()}')

    return best_eps, np.abs(pi_1_0 - best_policy).sum(), times
