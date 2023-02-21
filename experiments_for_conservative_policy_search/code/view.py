import matplotlib.pyplot as plt
import numpy as np
from Categories import RUN_TYPE

def draw(x, y, label, xlabel=None, ylabel=None, title=None, address=None, marker='o', alone=0):
    plt.scatter(x, y, alpha=0.5, marker=marker, label=label)
    plt.plot(x, y, alpha=0.5, ls='--')
    if alone == 1:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(loc='best')
        plt.savefig('../Plots/' + address)
        plt.clf()


def draw_plot(args, vals, naive_cost, CPS_cost, COPS_cost, UPS_cost):
    fig, ax = plt.subplots()

    n = 1 + len(CPS_cost) + len(COPS_cost) + len(UPS_cost)

    print(f'Naive = {naive_cost}')
    if len(naive_cost):
        ax.plot(vals, naive_cost, alpha=1, ls='--', marker='o', label=f'Naive', linewidth=n)

    print(f'CPS = {CPS_cost}')
    for i in range(len(CPS_cost)):
        if len(CPS_cost[i]):
            ax.plot(vals, CPS_cost[i], alpha=1, ls='--', marker='o', label=f'CPS_{i}', linewidth=n)
            n -= 1

    print(f'COPS = {COPS_cost}')
    for i in range(len(COPS_cost)):
        if len(COPS_cost[i]):
            ax.plot(vals, COPS_cost[i], alpha=1, ls='--', marker='o', label=f'COPS_{i}', linewidth=n)
            n -= 1

    print(f'UPS = {UPS_cost}')
    for i in range(len(UPS_cost)):
        if len(UPS_cost[i]):
            ax.plot(vals, UPS_cost[i], alpha=1, ls='--', marker='o', label=f'UPS_{i}', linewidth=n)
            n -= 1

    if args.var == 'epsilon':
        ax.set_xlabel('$\epsilon$', fontsize=28)
    else:
        ax.set_xlabel('$\iota$', fontsize=28)
    ax.set_ylabel('Cost', fontsize=28)


    ax.legend()

    fig.tight_layout()

    print(args.address)
    plt.savefig(f"{args.address}\\plot.pdf")

    plt.show()

def show_time(vals):

    print(f'average time: {np.round(np.mean(vals), 4)}')
    print(f'std: {np.round(np.std(vals), 4)}')
    print(f'max time: {np.round(np.max(vals), 4)}')
    print(f'min time: {np.round(np.min(vals), 4)}')

    plt.plot(vals)
    plt.show()