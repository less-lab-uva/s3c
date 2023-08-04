import argparse
from cycler import cycler
import os
import sys
from collections import defaultdict

import matplotlib
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob


label_map = {
    'carla_abstract': 'Entities + Road Structure',
    'carla_no_rel': 'Entities + Lanes',
    'carla_rsv': 'Entities + Lanes + Rel',
    'carla_rsv_filtered': 'Entities + Lanes + Rel Filtered',
    'carla_rsv_reldists': 'Entities + Lanes + Rel + Lane Spline',
    'carla_rsv_reldists_filtered': 'Entities + Lanes + Rel + Lane Spline Filtered',
    'carla_rsv_single': 'Entities + Lanes + Rel + Curvature',
    'carla_rsv_single_filtered': 'Entities + Lanes + Rel + Curvature Filtered',
    'carla_sem': 'Entities',
    'carla_sem_rel': 'Entities + Rel',
    'Label bin 0': 'All distinct labels',
    'Label bin 0.0001': 'Label bin $10^{-4}$ degrees',
    'Label bin 0.001': 'Label bin $10^{-3}$ degrees',
    'Label bin 0.01': 'Label bin $10^{-2}$ degrees',
    'Label bin 0.1': 'Label bin $10^{-1}$ degrees',
}
graphs_to_show = ['carla_sem', 'carla_no_rel', 'carla_sem_rel', 'carla_rsv', 'carla_abstract',
                    'Label bin 0', 'Label bin 0.0001', 'Label bin 0.001', 'Label bin 0.01', 'Label bin 0.1']


def custom_argparse(arg_string):
    parser = argparse.ArgumentParser(
        description="Meta Figure Visualizer"
    )
    parser.add_argument(
        "-i",
        "--input_path",
        type=Path,
        required=True,
        help="Location to load data set file."
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=Path,
        required=True,
        help="Directory to save figures"
    )
    return parser.parse_args(arg_string)


def meta_figure(arg_string):
    args = custom_argparse(arg_string)
    output_path = args.output_path/''
    os.makedirs(output_path, exist_ok=True)
    num_frames = None
    perc_failures_unseen_dict = defaultdict(list)
    num_cluster_dict = defaultdict(list)
    total_mse_list = []
    x = [0.8]
    for x_index, value in enumerate(x):
        glob_path = f'{args.input_path}/{value:.1f}_*/'
        folders = glob.glob(glob_path)
        if len(folders) != 1:
            raise ValueError(f"Could not find folder for value {value}")
        folder = folders[0]
        for graph_type in sorted(os.listdir(folder)):
            if 'filtered' in graph_type:
                continue
            data_folder = f'{folder}{graph_type}/'
            seen_stats = pd.read_csv(f'{data_folder}seen_stats.csv')
            num_cluster_dict[graph_type].append(seen_stats['num_clusters'].values[0])
            perc_failures_unseen_dict[graph_type].append(1-seen_stats['seen_perc'][0])
            if x_index == len(total_mse_list):
                total_mse_list.append(seen_stats['total_mse'].values[0])
        label_file = f'{folder}carla_abstract/label_groups.csv'
        label_groups = pd.read_csv(label_file)
        if num_frames is None:
            num_frames = label_groups['num_frames'].values[0]
        for precision in [0, 1e-4, 1e-3, 1e-2, 1e-1]:
            perc_failures_unseen_dict[f'Label bin {precision}'].append(1-label_groups.loc[np.isclose(label_groups.precision, precision)]['seen_perc'].values[0])
            num_cluster_dict[f'Label bin {precision}'].append(label_groups.loc[np.isclose(label_groups.precision, precision)]['num_clusters'].values[0])

    eighty_index = [index for index, val in enumerate(x) if abs(val - 0.8) < 1e-5][0]

    plt.rcParams['axes.prop_cycle'] = cycler('color', matplotlib.colormaps['tab20'].colors)

    fig = plt.figure()
    ax = fig.gca()
    for index, graph_type in enumerate(perc_failures_unseen_dict):
        ax.scatter(num_cluster_dict[graph_type], perc_failures_unseen_dict[graph_type], color=f'C{index}', label=f'{graph_type}')
    legend = ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax.set_xlabel('Number of clusters Train+Test (lower is better)')
    ax.set_ylabel('Fraction of Failures that are Unseen (higher is better)')
    fig.savefig(f'{output_path}/num_clusters.png', bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    ax = fig.gca()
    for index, graph_type in enumerate(perc_failures_unseen_dict):
        ax.scatter(num_cluster_dict[graph_type][eighty_index], perc_failures_unseen_dict[graph_type][eighty_index], color=f'C{index}',
                   label=f'{graph_type}')
    legend = ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax.set_xlabel('Number of clusters Train+Test (lower is better)')
    ax.set_ylabel('Fraction of Failures that are Unseen (higher is better)')
    fig.savefig(f'{output_path}/num_clusters_80_20.png', bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close(fig)

    plt.rcParams['axes.prop_cycle'] = cycler('color', matplotlib.colors.TABLEAU_COLORS)
    fig = plt.figure()
    ax = fig.gca()
    for index, graph_type in enumerate(graphs_to_show):
        label = label_map[graph_type] if graph_type in label_map else graph_type
        ax.scatter(num_cluster_dict[graph_type][eighty_index],
                   100 * perc_failures_unseen_dict[graph_type][eighty_index],
                   color=f'C{index}',
                   label=f'{label}',
                   marker='s' if 'Label' in graph_type else 'o')
    ax.scatter(num_frames, 100, color='k', label='All Data Unique\n(Trivial Solution)', marker='*')
    legend = ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax.set_xlabel('Number of clusters Train+Test')
    ax.set_ylabel('Percentage of Failures that are not Covered')
    fig.savefig(f'{output_path}/num_clusters_80_20_trivial.png', bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    ax = fig.gca()
    for index, graph_type in enumerate(graphs_to_show):
        label = label_map[graph_type] if graph_type in label_map else graph_type
        print(label)
        print('Total Clusters', num_cluster_dict[graph_type][eighty_index])
        print(f'PFNC {100 * perc_failures_unseen_dict[graph_type][eighty_index]}%')
        print()
        ax.scatter(num_cluster_dict[graph_type][eighty_index],
                   100 * perc_failures_unseen_dict[graph_type][eighty_index],
                   color=f'C{index}',
                   label=f'{label}',
                   marker='s' if 'Label' in graph_type else 'o')
    ax.scatter(num_frames, 100, color='k', label='All Data Unique\n(Trivial Solution)', marker='*')
    ax.hlines(20, 0, num_frames, linestyles='dashed', color='k', label='Random Baseline')
    legend = ax.legend()
    ax.set_xlabel('Total Number of Equivalence Classes')
    ax.set_ylabel('Percentage of Failures Not Covered (PFNC)')
    fig.savefig(f'{output_path}/num_clusters_80_20_trivial_inner_legend.png', bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    ax = fig.gca()
    for index, graph_type in enumerate(graphs_to_show):
        label = label_map[graph_type] if graph_type in label_map else graph_type
        ax.scatter(num_cluster_dict[graph_type][eighty_index],
                   100 * perc_failures_unseen_dict[graph_type][eighty_index],
                   color=f'C{index}',
                   label=f'{label}',
                   marker='s' if 'Label' in graph_type else 'o')
    ax.scatter(num_frames, 100, color='k', label='All Data Unique\n(Trivial Solution)', marker='*')
    legend = ax.legend(bbox_to_anchor=(0, -.1), loc='upper left', ncols=2)
    # ax.set_xlabel('Number of clusters Train+Test (lower is better)')
    # ax.set_ylabel('Percentage of Failures that are not Covered (higher is better)')
    ax.set_xlabel('Number of clusters Train+Test')
    ax.set_ylabel('Percentage of Failures that are not Covered')
    fig.savefig(f'{output_path}/num_clusters_80_20_trivial_legend_below.png', bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    meta_figure(sys.argv[1:])
