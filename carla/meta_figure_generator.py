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
    for folder in os.listdir(args.input_path):
        # print(folder)
        graph_df = pd.DataFrame({
            'file': pd.Series(dtype='string'),
            'perc_failures_unseen': pd.Series(dtype='float'),
            'perc_unseen_tests_fail': pd.Series(dtype='float'),
            'mix': pd.Series(dtype='float'),
            'num_clusters': pd.Series(dtype='int'),
            'unseen_mse': pd.Series(dtype='float'),
            'seen_mse': pd.Series(dtype='float'),
            'max_squared_error': pd.Series(dtype='float'),
            'total_mse': pd.Series(dtype='float')
        })
        for graph_type in sorted(os.listdir(f'{args.input_path}/{folder}')):
            if 'filtered' in graph_type:
                continue
            data_folder = f'{args.input_path}/{folder}/{graph_type}/'
            seen_stats = pd.read_csv(f'{data_folder}seen_stats.csv')
            # print(f'{data_folder}seen_stats.csv')
            # print(seen_stats)
            perc_failures_unseen = 1-seen_stats['seen_perc'][0]
            perc_unseen_tests_fail = seen_stats['perc_unseen_tests_fail'][0]
            mix = perc_failures_unseen * perc_unseen_tests_fail
            graph_df.loc[len(graph_df.index)] = [graph_type, perc_failures_unseen, perc_unseen_tests_fail, mix,
                                                 seen_stats['num_clusters'].values[0], seen_stats['unseen_mse'].values[0],
                                                 seen_stats['seen_mse'].values[0], seen_stats['max_squared_error'].values[0],
                                                 seen_stats['total_mse'].values[0]]
        label_file = f'{args.input_path}/{folder}/carla_abstract/label_groups.csv'
        label_groups = pd.read_csv(label_file)
        if num_frames is None:
            num_frames = label_groups['num_frames'].values[0]
        # print(label_groups)
        for precision in [0, 1e-4, 1e-3, 1e-2, 1e-1]:
            perc_failures_unseen = 1-label_groups.loc[np.isclose(label_groups.precision, precision)]['seen_perc'].values[0]
            perc_unseen_tests_fail = label_groups.loc[np.isclose(label_groups.precision, precision)]['perc_unseen_tests_fail'].values[0]
            mix = perc_failures_unseen * perc_unseen_tests_fail
            graph_df.loc[len(graph_df.index)] = [f'Label bin {precision}',
                                                 perc_failures_unseen,
                                                 perc_unseen_tests_fail,
                                                 mix,
                                                 label_groups.loc[np.isclose(label_groups.precision, precision)]['num_clusters'].values[0],
                                                 label_groups.loc[np.isclose(label_groups.precision, precision)]['unseen_mse'].values[0],
                                                 label_groups.loc[np.isclose(label_groups.precision, precision)]['seen_mse'].values[0],
                                                 label_groups.loc[np.isclose(label_groups.precision, precision)]['max_squared_error'].values[0],
                                                 label_groups.loc[np.isclose(label_groups.precision, precision)]['total_mse'].values[0]]
        graph_df.to_csv(f'{output_path}/{folder}.csv')
    perc_failures_unseen_dict = defaultdict(list)
    perc_unseen_tests_fail_dict = defaultdict(list)
    num_cluster_dict = defaultdict(list)
    unseen_mse_dict = defaultdict(list)
    seen_mse_dict = defaultdict(list)
    max_mse_dict = defaultdict(list)
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
            perc_failures_unseen_dict[graph_type].append(1-seen_stats['seen_perc'][0])
            perc_unseen_tests_fail_dict[graph_type].append(seen_stats['perc_unseen_tests_fail'][0])
            num_cluster_dict[graph_type].append(seen_stats['num_clusters'].values[0])
            unseen_mse_dict[graph_type].append(seen_stats['unseen_mse'].values[0])
            seen_mse_dict[graph_type].append(seen_stats['seen_mse'].values[0])
            max_mse_dict[graph_type].append(seen_stats['max_squared_error'].values[0])
            if x_index == len(total_mse_list):
                total_mse_list.append(seen_stats['total_mse'].values[0])
        label_file = f'{folder}carla_abstract/label_groups.csv'
        label_groups = pd.read_csv(label_file)
        for precision in [0, 1e-4, 1e-3, 1e-2, 1e-1]:
            perc_failures_unseen_dict[f'Label bin {precision}'].append(1-label_groups.loc[np.isclose(label_groups.precision, precision)]['seen_perc'].values[0])
            perc_unseen_tests_fail_dict[f'Label bin {precision}'].append(label_groups.loc[np.isclose(label_groups.precision, precision)]['perc_unseen_tests_fail'].values[0])
            num_cluster_dict[f'Label bin {precision}'].append(label_groups.loc[np.isclose(label_groups.precision, precision)]['num_clusters'].values[0])
            unseen_mse_dict[f'Label bin {precision}'].append(label_groups.loc[np.isclose(label_groups.precision, precision)]['unseen_mse'].values[0])
            seen_mse_dict[f'Label bin {precision}'].append(label_groups.loc[np.isclose(label_groups.precision, precision)]['seen_mse'].values[0])
            max_mse_dict[f'Label bin {precision}'].append(label_groups.loc[np.isclose(label_groups.precision, precision)]['max_squared_error'].values[0])
    eighty_index = [index for index, val in enumerate(x) if abs(val - 0.8) < 1e-5][0]
    x = np.array(x)
    plt.rcParams['axes.prop_cycle'] = cycler('color', matplotlib.colormaps['tab20'].colors)
    fig = plt.figure()
    ax = fig.gca()
    for index, graph_type in enumerate(perc_failures_unseen_dict):
        ax.plot(x, perc_failures_unseen_dict[graph_type], marker='.', linestyle='-', color=f'C{index}', label=f'{graph_type}: % failures unseen')
    for index, graph_type in enumerate(perc_failures_unseen_dict):
        ax.plot(x, perc_unseen_tests_fail_dict[graph_type], marker='*', linestyle='--', color=f'C{index}', label=f'{graph_type}: % of unseen tests that fail')
    legend = ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax.set_xlabel('% of data used in training')
    ax.set_ylabel('Metric value [0,1]')
    fig.savefig(f'{output_path}/perc_fail_and_perc_unseen.png', bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close(fig)
    
    fig = plt.figure()
    ax = fig.gca()
    for index, graph_type in enumerate(perc_failures_unseen_dict):
        ax.plot(x, perc_failures_unseen_dict[graph_type], marker='.', linestyle='-', color=f'C{index}', label=f'{graph_type}')
    ax.plot(x, 1-x, marker='.', linestyle='-', color='k', label='Theoretical random')
    legend = ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax.set_xlabel('% of data used in training')
    ax.set_ylabel('Fraction of Failures Unseen (higher is better)')
    fig.savefig(f'{output_path}/perc_fail.png', bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    ax = fig.gca()
    for index, graph_type in enumerate(perc_failures_unseen_dict):
        ax.plot(x, perc_unseen_tests_fail_dict[graph_type], marker='.', linestyle='-', color=f'C{index}', label=f'{graph_type}')
    legend = ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax.set_xlabel('% of data used in training')
    ax.set_ylabel('Fraction of Unseen Tests that Fail (higher is better)')
    fig.savefig(f'{output_path}/perc_unseen_fail.png', bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    ax = fig.gca()
    for index, graph_type in enumerate(perc_failures_unseen_dict):
        ax.plot(x, [a * b for a, b in zip(perc_unseen_tests_fail_dict[graph_type], perc_failures_unseen_dict[graph_type])], marker='.', linestyle='-', color=f'C{index}', label=f'{graph_type}')
    legend = ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax.set_xlabel('% of data used in training')
    ax.set_ylabel('Mix: Product of two metrics (higher is better)')
    fig.savefig(f'{output_path}/mix.png', bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    ax = fig.gca()
    for index, graph_type in enumerate(perc_failures_unseen_dict):
        ax.scatter(perc_unseen_tests_fail_dict[graph_type], perc_failures_unseen_dict[graph_type], color=f'C{index}', label=f'{graph_type}')
    legend = ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax.set_xlabel('Fraction of Unseen Tests that Fail (higher is better)')
    ax.set_ylabel('Fraction of Failures that are Unseen (higher is better)')
    fig.savefig(f'{output_path}/mix_scatter.png', bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close(fig)

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
        print('total clusters', num_cluster_dict[graph_type][eighty_index])
        print('Failure perc', 100 * perc_failures_unseen_dict[graph_type][eighty_index])
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

    plt.rcParams['axes.prop_cycle'] = cycler('color', matplotlib.colormaps['tab20'].colors)
    fig = plt.figure()
    ax = fig.gca()
    legend_patches = []
    for index, graph_type in enumerate(perc_failures_unseen_dict):
        # print('box data', graph_type)
        # print(perc_failures_unseen_dict[graph_type])
        bp = ax.boxplot(perc_failures_unseen_dict[graph_type],
                        positions=[num_cluster_dict[graph_type][0]],
                        vert=True,
                        widths=num_frames/100,
                        patch_artist=True,
                        manage_ticks=False)
        legend_patches.append(matplotlib.patches.Patch(color=f'C{index}', label=f'{graph_type}'))
        for patch in bp['boxes']:
            patch.set_facecolor(f'C{index}')
    legend = ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax.set_xlabel('Number of clusters Train+Test (lower is better)')
    ax.set_ylabel('Fraction of Failures that are Unseen (higher is better)')
    fig.savefig(f'{output_path}/num_clusters_box.png', bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    ax = fig.gca()
    for index, graph_type in enumerate(perc_failures_unseen_dict):
        ax.scatter(num_cluster_dict[graph_type], perc_failures_unseen_dict[graph_type], color=f'C{index}', label=f'{graph_type}')
    ax.scatter(num_frames, 1.0, color='k', label='All Data Unique (Trivial Solution)')
    legend = ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax.set_xlabel('Number of clusters Train+Test (lower is better)')
    ax.set_ylabel('Fraction of Failures that are Unseen (higher is better)')
    fig.savefig(f'{output_path}/num_clusters_trivial.png', bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    ax = fig.gca()
    for index, graph_type in enumerate(perc_failures_unseen_dict):
        ax.scatter(seen_mse_dict[graph_type], unseen_mse_dict[graph_type], color=f'C{index}', label=f'{graph_type}')
    legend = ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    # set limits to be same on both axes
    upper_limit = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.set_xlim(0, upper_limit)
    ax.set_ylim(0, upper_limit)
    ax.set_xlabel('MSE of Seen Tests (lower is better)')
    ax.set_ylabel('MSE of Unseen Tests (higher is better)')
    fig.savefig(f'{output_path}/mse_unseen_vs_seen.png', bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    ax = fig.gca()
    for index, graph_type in enumerate(perc_failures_unseen_dict):
        ax.plot(x, unseen_mse_dict[graph_type], marker='.', linestyle='-', color=f'C{index}', label=f'Unseen MSE {graph_type}')
        ax.plot(x, seen_mse_dict[graph_type], marker='.', linestyle='--', color=f'C{index}',
                label=f'Seen MSE {graph_type}' if index == 0 else '_none')
    ax.plot(x, total_mse_list, label='MSE over all Tests', color='k')
    legend = ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax.set_xlabel('% of data used in training')
    ax.set_ylabel('MSE')
    fig.savefig(f'{output_path}/mse_by_split.png', bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    ax = fig.gca()
    for index, graph_type in enumerate(perc_failures_unseen_dict):
        ax.plot(x, np.array(unseen_mse_dict[graph_type]) / np.array(seen_mse_dict[graph_type]), marker='.', linestyle='-', color=f'C{index}',
                label=f'Unseen MSE/Seen MSE {graph_type}')
    legend = ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax.set_xlabel('% of data used in training')
    ax.set_ylabel('Unseen to Seen MSE Ratio (higher is better)')
    fig.savefig(f'{output_path}/mse_by_split_ratio.png', bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    ax = fig.gca()
    for index, graph_type in enumerate(perc_failures_unseen_dict):
        ax.scatter(np.array(unseen_mse_dict[graph_type]) / np.array(seen_mse_dict[graph_type]), perc_failures_unseen_dict[graph_type], marker='.',
                linestyle='-', color=f'C{index}',
                label=f'{graph_type}')
    legend = ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax.set_xlabel('Unseen to Seen MSE Ratio (higher is better)')
    ax.set_ylabel('Fraction of Unseen Tests that Fail (higher is better)')
    fig.savefig(f'{output_path}/fail_perc_vs_mse_ratio.png', bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    ax = fig.gca()
    for index, graph_type in enumerate(perc_failures_unseen_dict):
        ax.scatter(np.array(unseen_mse_dict[graph_type]),
                   perc_failures_unseen_dict[graph_type], marker='.',
                   linestyle='-', color=f'C{index}',
                   label=f'{graph_type}')
    legend = ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax.set_xlabel('Unseen MSE (higher is better)')
    ax.set_ylabel('Fraction of Unseen Tests that Fail (higher is better)')
    fig.savefig(f'{output_path}/fail_perc_vs_unseen_mse.png', bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close(fig)

    for x_index, i in enumerate(x):
        fig = plt.figure()
        ax = fig.gca()
        for index, graph_type in enumerate(perc_failures_unseen_dict):
            ax.scatter(seen_mse_dict[graph_type][x_index], unseen_mse_dict[graph_type][x_index], color=f'C{index}', label=f'{graph_type}')
        ax.axvline(total_mse_list[x_index], label='MSE over all Tests', color='k')
        ax.axhline(total_mse_list[x_index], label='_none', color='k')
        legend = ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        # set limits to be same on both axes
        upper_limit = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.set_xlim(0, upper_limit)
        ax.set_ylim(0, upper_limit)
        ax.set_xlabel('MSE of Seen Tests (lower is better)')
        ax.set_ylabel('MSE of Unseen Tests (higher is better)')
        fig.savefig(f'{output_path}/mse_unseen_vs_seen_{i:.1f}.png', bbox_extra_artists=(legend,), bbox_inches='tight')
        plt.close(fig)

    for x_index, i in enumerate(x):
        fig = plt.figure()
        ax = fig.gca()
        for index, graph_type in enumerate(perc_failures_unseen_dict):
            ax.scatter(unseen_mse_dict[graph_type][x_index], perc_failures_unseen_dict[graph_type][x_index], color=f'C{index}', label=f'{graph_type}')
        ax.axvline(total_mse_list[x_index], label='MSE over all Tests', color='k')
        legend = ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        ax.set_xlabel('MSE of Unseen Tests (higher is better)')
        ax.set_ylabel('Fraction of Unseen Tests that Fail (higher is better)')
        fig.savefig(f'{output_path}/perc_fail_unseen_vs_unseen_mse_{i:.1f}.png', bbox_extra_artists=(legend,), bbox_inches='tight')
        plt.close(fig)

    fig = plt.figure()
    ax = fig.gca()
    markers = ['.', '*', '^', '1', 's', 'P', '+', 'x', 'D']
    for x_index, i in enumerate(x):
        for index, graph_type in enumerate(perc_failures_unseen_dict):
            if index == x_index:
                label = f'{graph_type} ({i:.1f})'
            elif (index >= len(x) and x_index == len(x) - 1):
                label = f'{graph_type}'
            else:
                label = '_none'
            ax.scatter(seen_mse_dict[graph_type][x_index], unseen_mse_dict[graph_type][x_index], color=f'C{index}', label=label,
                       marker=markers[x_index])
    legend = ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    # set limits to be same on both axes
    upper_limit = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.set_xlim(0, upper_limit)
    ax.set_ylim(0, upper_limit)
    ax.set_xlabel('MSE of Seen Tests (lower is better)')
    ax.set_ylabel('MSE of Unseen Tests (higher is better)')
    fig.savefig(f'{output_path}/mse_unseen_vs_seen_markers.png', bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    ax = fig.gca()
    for x_index, i in enumerate(x):
        for index, graph_type in enumerate(perc_failures_unseen_dict):
            if index == x_index:
                label = f'{graph_type} ({i:.1f})'
            elif (index >= len(x) and x_index == len(x) - 1):
                label = f'{graph_type}'
            else:
                label = '_none'
            ax.scatter(unseen_mse_dict[graph_type][x_index] / total_mse_list[x_index],
                       perc_failures_unseen_dict[graph_type][x_index],
                       color=f'C{index}', label=label, marker=markers[x_index])
    legend = ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax.set_xlabel('Unseen MSE / MSE on whole (higher is better)')
    ax.set_ylabel('Fraction of Unseen Tests that Fail (higher is better)')
    fig.savefig(f'{output_path}/perc_fail_unseen_vs_unseen_mse.png', bbox_extra_artists=(legend,),
                bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    meta_figure(sys.argv[1:])
