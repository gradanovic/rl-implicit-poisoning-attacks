import environment
import numpy as np
import algorithms
import utils
import scipy.stats as sp
import test
import math
from Categories import RUN_TYPE, ALGORITHM, OPTIMIZER, STATE_OCCUPANCY
import argparse


# Output of all below functions: env, pi_1_ref, pi_1_naive, pi_2_target, ergodicity, inf (showing infeasibility cost)

# Old Simple Chain
###############################################
def old_single_chain():
    # Create the Environment
    env = environment.create_old_chain_env(alpha=1, beta=-1, P1=0.9, P2=0.6, gamma=0.9)
    # define the initial policy
    pi_1_ref = np.array([[1, 0], [1, 0], [1, 0], [1, 0]])
    # define the naive policy
    pi_1_naive = np.array([[0, 1], [0, 1], [0, 1], [0, 1]])
    # define the target policy
    pi_2_target = np.array([[0, 1], [0, 1], [0, 1], [0, 1]])

    return env, pi_1_ref, pi_1_naive, pi_2_target, True, 10
###############################################

# Old Navigation
###############################################
def old_navigation():
    # create the environment
    env = environment.create_old_navigation_chain(alpha=5, beta=-5, P1=0.9, P2=0.6, gamma=0.9)
    # define the initial policy
    pi_1_ref = np.zeros((env.n_states, env.n_actions))
    pi_1_ref[:, 0] = 1
    # define the naive policy
    pi_1_naive = np.zeros((env.n_states, env.n_actions))
    pi_1_naive[:, 1] = 1
    # define the target policy
    pi_2_target = np.zeros((env.n_states, env.n_actions))
    pi_2_target[:, 1] = 1

    return env, pi_1_ref, pi_1_naive, pi_2_target, True, 18
###############################################


# New Simple Chain - New Navigation
###############################################
def navigation():
    # create the environment
    env = environment.create_navigation(p=0.9, gamma=0.9)
    # define the initial policy
    pi_1_ref = np.zeros((env.n_states, env.n_actions))
    pi_1_ref[:, 0] = 1
    # define the naive policy
    pi_1_naive = np.zeros((env.n_states, env.n_actions))
    pi_1_naive[:, 1] = 1
    # define the target policy
    pi_2_target = np.zeros((env.n_states, env.n_actions))
    pi_2_target[:, 1] = 1

    return env, pi_1_ref, pi_1_naive, pi_2_target, True, 20
###############################################


# Single Item Inventory Control Navigation
###############################################
def single_inventory():
    # create the environment
    env = environment.create_inventory_control(M=10, gamma=0.9, reject=True)
    # define the initial policy
    # Be careful !! pi_1_ref has to be fully stochastic !!
    pi_1_ref = np.ones((env.n_states, env.n_actions)) / env.n_actions
    # define the target policy
    k = 7
    pi_2_target = np.zeros((env.n_states, env.n_actions))
    for i in range(env.n_states):
        pi_2_target[i, max(0, k - i)] = 1
    # define the naive policy
    pi_1_naive = np.zeros((env.n_states, env.n_actions))
    pi_1_naive[:, k] = 1

    return env, pi_1_ref, pi_1_naive, pi_2_target, False, 20
###############################################


# Select the environment given the inputs, and execute the code
def run_experiment(args, vals, CPS_vals, COPS_vals, UPS_vals):
    if args.env == 'navigation':
        env, pi_1_ref, pi_1_naive, pi_2_target, ergodicity, inf = navigation()
    if args.env == 'inventory':
        env, pi_1_ref, pi_1_naive, pi_2_target, ergodicity, inf = single_inventory()
    if args.env == 'old_navigation':
        env, pi_1_ref, pi_1_naive, pi_2_target, ergodicity, inf = old_navigation()

    test.grid_search(args, vals, CPS_vals, COPS_vals, UPS_vals, env, pi_1_ref, pi_1_naive, pi_2_target, ergodicity, inf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experimental Settings')
    # env: Select your desired environment from {inventory, navigation}
    parser.add_argument('--env', type=str, default='navigation', help='navigation, inventory, old_navigation')
    # var: Select the variable in different experiments from {epsilon, influence}
    parser.add_argument('--var', type=str, default='influence', help='epsilon, influence')
    # Select the algorithms that you want to evaluate by changing corresponding bit to 1, otherwise, 0.
    # MASK = (NAIVE, CPS, COPS, UPS)
    parser.add_argument('--alg', type=str, default='1111')
    # Select the number of iterations for main loop of the algorithms
    parser.add_argument('--n_iter', type=int, default=200)
    # Select the number of iterations for computing bellman equation
    parser.add_argument('--util_iter', type=int, default=200)
    # Select the default value of epsilon when the variable is influence
    parser.add_argument('--eps', type=float, default=0.05)
    # Set this parameter to True, if you want to evaluate the time of executing of an algorithm
    parser.add_argument('--time_eval', type=bool, default=False)
    # Set the number of runs for evaluating an algorithm
    parser.add_argument('--time_eval_iter', type=int, default=10)
    # Select the final destination where the results will be stored to.
    parser.add_argument('--address', type=str, default='..\\new_data\\test')
    args = parser.parse_args()

    ## Algorithms : Naive, CPS, COPS, UPS
    vals = list(map(float, input('####\nenter values for the desired variable:\n').split()))
    CPS_vals, COPS_vals, UPS_vals = [], [], []
    if args.alg[1] == '1':
        CPS_vals = input('####\nenter (lam,delta) values for CPS separated by space:\n').split()
    if args.alg[2] == '1':
        COPS_vals = input('####\nenter delta values for COPS separated by space:\n').split()
    if args.alg[3] == '1':
        UPS_vals = input('####\nenter lam values for UPS separated by space:\n').split()

    run_experiment(args, vals, CPS_vals, COPS_vals, UPS_vals)
