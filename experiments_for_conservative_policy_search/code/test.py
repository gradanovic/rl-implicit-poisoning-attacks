## This files consists of codes for executing the algorithms on different values

import algorithms
import numpy as np
import view
from Categories import RUN_TYPE, ALGORITHM
import os
import shutil
import environment
import re

## All reported times from different runs
iterations_time = []

# Check if the output of algorithm achieves epsilon which is large enough.
def check_output(output, epsilon, inf):
    eps, cost = output[0], output[1]
    iterations_time.extend(output[2])
    if eps < epsilon:
        return inf
    else:
        return cost

# Save the results of each algorithm
def save(args, vals, naive_cost, CPS_cost, COPS_cost, UPS_cost):
    PATH = args.address
    os.makedirs(PATH, exist_ok=True)

    np.save(PATH + f'\\{args.var}.npy', vals)

    if len(naive_cost):
        np.save(PATH + f'\\naive_cost.npy', naive_cost)

    for i in range(len(CPS_cost)):
        if len(CPS_cost[i]):
            np.save(PATH + f'\\CPS_cost_{i}.npy', CPS_cost[i])

    for i in range(len(COPS_cost)):
        if len(COPS_cost[i]):
            np.save(PATH + f'\\COPS_cost_{i}.npy', COPS_cost[i])

    for i in range(len(UPS_cost)):
        if len(UPS_cost[i]):
            np.save(PATH + f'\\UPS_cost_{i}.npy', UPS_cost[i])


# Execute each algorithm with the given hyper-paramteres
def grid_search(args, vals, CPS_vals, COPS_vals, UPS_vals, env: environment.Environment, pi_1_ref, pi_1_naive,
                pi_2_target, ergodicity, inf):
    naive_cost = []
    CPS_cost = [[] for _ in range(len(CPS_vals))]
    COPS_cost = [[] for _ in range(len(COPS_vals))]
    UPS_cost = [[] for _ in range(len(UPS_vals))]

    N = 1 if not args.time_eval else args.time_eval_iter

    for _ in range(N):
        for val in vals:
            if args.var == 'epsilon':
                epsilon = val
            else:
                epsilon = args.eps
                env.change_influence(val, pi_1_ref)

            if args.alg[0] == '1':
                print('Naive')
                naive_cost.append(check_output(
                    algorithms.heuristic_algorithm(env, pi_1_ref, pi_1_naive, pi_2_target, epsilon=epsilon,
                                                   delta=None, lam=None, algorithm=ALGORITHM.NAIVE, n_iter=args.n_iter,
                                                   util_iter=args.util_iter, ergodicity=ergodicity), epsilon, inf))

            if args.alg[1] == '1':
                print('CPS')
                id = 0
                for conf in CPS_vals:
                    res = re.split('[,()]', conf)
                    lam, delta = float(res[1]), float(res[2])
                    CPS_cost[id].append(check_output(
                        algorithms.heuristic_algorithm(env, pi_1_ref, pi_1_naive, pi_2_target, epsilon=epsilon,
                                                       delta=delta, lam=lam, algorithm=ALGORITHM.CPS, n_iter=args.n_iter,
                                                       util_iter=args.util_iter, ergodicity=ergodicity), epsilon, inf))
                    id += 1

            if args.alg[2] == '1':
                print('COPS')
                id = 0
                for delta in COPS_vals:
                    COPS_cost[id].append(check_output(
                        algorithms.heuristic_algorithm(env, pi_1_ref, pi_1_naive, pi_2_target, epsilon=epsilon,
                                                       delta=float(delta), lam=None, algorithm=ALGORITHM.COPS,
                                                       n_iter=args.n_iter, util_iter=args.util_iter, ergodicity=ergodicity), epsilon, inf))
                    id += 1

            if args.alg[3] == '1':
                print('UPS')
                id = 0
                for lam in UPS_vals:
                    UPS_cost[id].append(check_output(
                        algorithms.heuristic_algorithm(env, pi_1_ref, pi_1_naive, pi_2_target, epsilon=epsilon,
                                                       delta=None, lam=float(lam), algorithm=ALGORITHM.UPS,
                                                       n_iter=args.n_iter, util_iter=args.util_iter, ergodicity=ergodicity), epsilon, inf))
                    id += 1

    # Evaluate the spent time
    if args.time_eval:
        view.show_time(iterations_time)

    # Save results
    else:
        save(args, vals, naive_cost, CPS_cost, COPS_cost, UPS_cost)

        # print results for the user
        print(naive_cost)
        print(CPS_cost)
        print(COPS_cost)
        print(UPS_cost)

        # draw the corresponding plot
        view.draw_plot(args, vals, naive_cost, CPS_cost, COPS_cost, UPS_cost)
