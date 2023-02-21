## This folder is used to create plots for the published paper
## You can customize the functions as you want

import numpy as np
import matplotlib.pyplot as plt
import argparse
import matplotlib


def set_max(x, mx):
    tmp = [min(val, mx) for val in x]
    return tmp


def plot(args):
    main_dir = args.address + '\\'

    if args.legend:
        plt.rcParams.update({'font.size': 12})
    else:
        plt.rcParams.update({'font.size': 25})
    fig, ax = plt.subplots()

    variables = np.load(main_dir + args.var + '.npy')
    # variables = [0]
    if args.mode == 'general':
        naive_cost = np.load(main_dir + 'naive_cost.npy')
        CPS_cost = np.load(main_dir + 'CPS_cost_0.npy')
        COPS_cost = np.load(main_dir + 'COPS_cost_0.npy')
        UPS_cost = np.load(main_dir + 'UPS_cost_0.npy')

        markersize = 9
        max_val = 20
        print('cps:', CPS_cost)
        ax.plot(variables, set_max(CPS_cost, max_val), alpha=1, ls='--', marker='s', label='CPS', linewidth=1, markersize=markersize)
        print('cops:', COPS_cost)
        ax.plot(variables, set_max(COPS_cost, max_val), alpha=1, ls='--', marker='o', label='COPS', linewidth=1, markersize=markersize)
        print('naive:', naive_cost)
        ax.plot(variables, set_max(naive_cost, max_val), alpha=1, ls='--', marker='v', label='Naive', linewidth=1, markersize=markersize)
        print('ups:', UPS_cost)
        ax.plot(variables, set_max(UPS_cost, max_val), alpha=1, ls='--', marker='P', label='UPS', linewidth=1, markersize=markersize)

    else:
        costs = []
        for i in range(5):
            # costs.append([0])
            costs.append(np.load(main_dir + f'CPS_cost_{i}.npy'))
        labels = ['$\lambda = 20, \delta = 0.01$', '$\lambda = 5, \delta = 0.01$', '$\lambda = 75, \delta = 0.01$',
                  '$\lambda = 20, \delta = 0.1$', '$\lambda = 20, \delta = 1$']
        linewidths = np.array([1, 1, 1, 1, 1]) * 2
        for i in range(5):
            print(f'cost_{i}:', costs[i])
            ax.plot(variables, set_max(costs[i], 20), alpha=1, ls='--', marker='s', markersize=8, label=labels[i], linewidth=linewidths[i])

    if args.env == 'navigation':
        if args.var == 'epsilon':
            ax.set_xticks([0.01, 0.03, 0.05, 0.07, 0.09])
        else:
            ax.set_xticks([0.8, 0.85, 0.9, 0.95, 1])
        plt.yticks([10, 12, 14, 16, 18], [10, 12, 14, 16, 'inf'])
    else:
        if args.var == 'epsilon':
            ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
        else:
            ax.set_xticks([0.6, 0.7, 0.8, 0.9, 1])
        plt.yticks([5, 9, 13, 17, 20], [5, 9, 13, 17, 'inf'])

    if args.var == 'epsilon':
        ax.set_xlabel('$\epsilon$', fontsize=28)
    else:
        ax.set_xlabel('$\iota$', fontsize=28)
    ax.set_ylabel('Cost', fontsize=28)

    fig.tight_layout()

    if args.legend:
        fig.tight_layout()

        legend = plt.legend(loc='upper center', ncol=3, fancybox=True, shadow=True, fontsize='xx-large')
        fig.canvas.draw()
        legend_bbox = legend.get_tightbbox(fig.canvas.get_renderer())
        legend_bbox = legend_bbox.transformed(fig.dpi_scale_trans.inverted())
        legend_fig, legend_ax = plt.subplots(figsize=(legend_bbox.width, legend_bbox.height))
        legend_squared = legend_ax.legend(
            *ax.get_legend_handles_labels(),
            bbox_to_anchor=(0, 0, 1, 1),
            bbox_transform=legend_fig.transFigure,
            frameon=True,
            fancybox=True,
            shadow=True,
            ncol=5,
            fontsize='xx-large',
        )
        legend_ax.axis('off')
        legend_fig.savefig(args.save_address,
                           bbox_inches='tight',
                           bbox_extra_artists=[legend_squared]
                           )
    else:
        plt.savefig(args.save_address)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experimental Settings')
    # Select the address of stored data
    parser.add_argument('--address', type=str, default='..\\new_data\\inventory_CPS_influence' )
    # Set this parameter to True, if you want to create the legend
    parser.add_argument('--legend', type=bool, default=False)
    # Select the environment
    parser.add_argument('--env', type=str, default='inventory', help='navigation, inventory, old_navigation')
    # Select the environment
    parser.add_argument('--var', type=str, default='influence', help='epsilon, influence')
    # Set this parameter to general, if you are plotting the results for different algorithms, otherwise, set it to sepecific
    parser.add_argument('--mode', type=str, default='specific', help='general, specific')
    # Select the address and the name of the plot
    parser.add_argument('--save_address', type=str, default='..\\new_plots\\new-inventory-CPS-influence.pdf')

    args = parser.parse_args()

    plot(args)
