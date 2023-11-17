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
    # 'Label bin 0': 'All distinct labels',
    'Label bin 0': 'Codomain Ground Truth',
    'Label bin 0.0001': 'Label bin $10^{-4}$ degrees',
    'Label bin 0.001': 'Label bin $10^{-3}$ degrees',
    'Label bin 0.01': 'Label bin $10^{-2}$ degrees',
    'Label bin 0.1': 'Label bin $10^{-1}$ degrees',
    'rss_1': '$\\Psi_{1}$ (PhysCov)',
    'rss_5': '$\\Psi_{5}$ (PhysCov)',
    'rss_10': '$\\Psi_{10}$ (PhysCov)',
    'rss_v1_10': '$\\Psi_{10}$V1 (PhysCov)',
    'rss_v2_10': '$\\Psi_{10}^{*}$ (PhysCov)',
    'carla_abstract_rss_10': 'ERS $\\times \\Psi_{10}$',
    'carla_no_rel_rss_10': 'EL $\\times \\Psi_{10}$',
    'carla_rsv_rss_10': 'ELR $\\times \\Psi_{10}$',
    'carla_sem_rel_rss_10': 'ER $\\times \\Psi_{10}$',
    'carla_sem_rss_10': 'E $\\times \\Psi_{10}$',

    'carla_abstract_rss_v1_10': 'ERS $\\times \\Psi_{10}$V1',
    'carla_no_rel_rss_v1_10': 'EL $\\times \\Psi_{10}$V1',
    'carla_rsv_rss_v1_10': 'ELR $\\times \\Psi_{10}$V1',
    'carla_sem_rel_rss_v1_10': 'ER $\\times \\Psi_{10}$V1',
    'carla_sem_rss_v1_10': 'E $\\times \\Psi_{10}$V1',

    'carla_abstract_rss_v2_10': 'ERS $\\times \\Psi_{10}$V2',
    'carla_no_rel_rss_v2_10': 'EL $\\times \\Psi_{10}$V2',
    'carla_rsv_rss_v2_10': 'ELR $\\times \\Psi_{10}$V2',
    'carla_sem_rel_rss_v2_10': 'ER $\\times \\Psi_{10}$V2',
    'carla_sem_rss_v2_10': 'E $\\times \\Psi_{10}$V2',

    'carla_rsv_time_2': 'Entities + Lanes + Rel $\\times$ 2 frames',
    'carla_rsv_time_3': 'ELR[3]',
    'carla_rsv_time_5': 'Entities + Lanes + Rel $\\times$ 5 frames',
    'carla_rsv_time_10': 'ELR[10]',
    'carla_abstract_time_2': 'ERS[2]',
    'carla_abstract_time_3': 'ERS[3]',
    'carla_abstract_time_5': 'ERS[5]',
    'carla_abstract_time_10': 'ERS[10]',
}
graphs_to_show = ['carla_sem', 'carla_no_rel', 'carla_sem_rel', 'carla_rsv', 'carla_abstract',
                  'Label bin 0',
                  'rss_10',
                  'rss_v2_10',
                  ]

def get_color(graph_type, index):
    # Guarantee that the graph-derived ones use the same color
    if 'carla_sem_rel' in graph_type:
        return 'C2'
    if 'carla_sem' in graph_type:
        return 'C0'
    if 'carla_no_rel' in graph_type:
        return 'C1'
    if 'carla_rsv' in graph_type:
        return 'C3'
    if 'carla_abstract' in graph_type:
        return 'C4'
    # Guarantee that if it not a graph-based one then
    return f'C{(index % 5)+5}'

def get_marker(graph_type):
    if 'rss' in graph_type:
        return '^'
    if 'Label' in graph_type:
        return '*'
    if 'time' in graph_type:
        if 'time_2' in graph_type:
            return '$2$'
        if 'time_5' in graph_type:
            return '$5$'
        if 'time_10' in graph_type:
            return '$0$'
    return 'o'

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


def normalize_random(clusters_list, random_list, bin_size=2000):
    end = int(max(clusters_list))
    bins = [i for i in range(0, end, bin_size)]
    x = []
    y = []
    for start, stop in zip(bins, bins[1:]):
        lower_index = min([i for i in range(len(clusters_list)) if clusters_list[i] >= start])
        upper_index = max([i for i in range(len(clusters_list)) if clusters_list[i] <= stop])
        temp_x = []
        temp_y = []
        for a, b in zip(clusters_list[lower_index:upper_index + 1],random_list[lower_index:upper_index+1]):
            if np.isnan(a) or np.isnan(b):
                continue
            temp_x.append(a)
            temp_y.append(b)
        x.append(np.mean(temp_x))
        y.append(np.mean(temp_y))
    return x, y


def meta_figure(arg_string):
    args = custom_argparse(arg_string)
    output_path = args.output_path/''
    os.makedirs(output_path, exist_ok=True)
    num_frames = None
    perc_failures_unseen_dict = defaultdict(list)
    perc_tests_unseen_dict = defaultdict(list)
    new_perc_failures_unseen_dict = defaultdict(list)
    new_perc_fail_classes_unseen_dict = defaultdict(list)
    num_cluster_dict = defaultdict(list)
    random_perc_failures_unseen_dict = defaultdict(list)
    random_num_cluster_dict = defaultdict(list)
    singleton_random_perc_failures_unseen_dict = defaultdict(list)
    singleton_random_num_cluster_dict = defaultdict(list)
    perc_consistent_class_dict = defaultdict(list)
    perc_possible_inconsistent_class_dict = defaultdict(list)
    perc_classes_with_failures_that_are_consistent_class_dict = defaultdict(list)
    total_mse_list = []
    x = [0.8]
    for x_index, value in enumerate(x):
        glob_path = f'{args.input_path}/{value:.1f}_*/'
        folders = glob.glob(glob_path)
        if len(folders) != 1:
            raise ValueError(f"Could not find folder for value {value}")
        folder = folders[0]
        for graph_type in sorted(os.listdir(folder)):
            try:
                if 'filtered' in graph_type:
                    continue
                if graph_type not in graphs_to_show:
                    continue
                data_folder = f'{folder}{graph_type}/'
                seen_stats = pd.read_csv(f'{data_folder}seen_stats.csv')
                num_cluster_dict[graph_type].append(seen_stats['num_clusters'].values[0])
                # perc_failures_unseen_dict[graph_type].append(1 - seen_stats['seen_perc'][0])
                perc_tests_unseen_dict[graph_type].append(1 - seen_stats['seen_perc'][0])
                perc_failures_unseen_dict[graph_type].append(1-seen_stats['fail_perc'][0])  # THIS IS CORRECT
                # print(graph_type)
                seen_stats['pnfnc'] = seen_stats.apply(lambda x: (x.num_new_failures - x.num_new_failures_covered) / x.num_new_failures if x.num_new_failures > 0 else np.nan, axis=1)
                seen_stats['pnfnc_classes'] = seen_stats.apply(lambda x: (x.num_new_failure_classes - x.num_new_failure_classes_covered) / x.num_new_failure_classes if x.num_new_failure_classes > 0 else np.nan, axis=1)
                new_perc_failures_unseen_dict[graph_type].append(seen_stats['pnfnc'])
                new_perc_fail_classes_unseen_dict[graph_type].append(seen_stats['pnfnc_classes'])
                seen_stats['perc_consistent_class'] = seen_stats.apply(lambda x: x.num_consistent_non_singleton_classes / x.num_non_singleton_classes, axis=1)
                seen_stats['perc_possible_inconsistent_class'] = seen_stats.apply(lambda x: (x.num_non_singleton_classes - x.num_consistent_non_singleton_classes) / min(x.num_clusters, x.total_failures), axis=1)
                perc_consistent_class_dict[graph_type].append(seen_stats['perc_consistent_class'])
                perc_possible_inconsistent_class_dict[graph_type].append(seen_stats['perc_possible_inconsistent_class'])
                seen_stats['perc_classes_with_failures_that_are_consistent'] = seen_stats.apply(lambda x: x.num_classes_with_all_failures / x.num_classes_with_failures, axis=1)
                perc_classes_with_failures_that_are_consistent_class_dict[graph_type] = seen_stats['perc_classes_with_failures_that_are_consistent']
                # perc_failures_unseen_dict[graph_type].append(seen_stats['unseen_test_fails'][0] / 100)
                # perc_failures_unseen_dict[graph_type].append(1e-2*(1-seen_stats['fail_perc'][0]) / (1-seen_stats['seen_perc'][0]) if (1-seen_stats['seen_perc'][0]) > 0 else 0)
                if x_index == len(total_mse_list):
                    total_mse_list.append(seen_stats['total_mse'].values[0])
            except:
                print(f'error on {graph_type}')
                pass
        label_file = f'{folder}carla_abstract/label_groups.csv'
        label_groups = pd.read_csv(label_file)
        if num_frames is None:
            num_frames = label_groups['num_frames'].values[0]
        label_pfnc = {num_clusters: 100 * (1 - label_groups[label_groups['num_clusters'] == num_clusters]['fail_perc'].mean()) for num_clusters in label_groups['num_clusters'].values.tolist()}
        label_pnfnc = {row.num_clusters: 100 * (row.num_new_failures - row.num_new_failures_covered) / row.num_new_failures if row.num_new_failures > 0 else np.nan for index, row in label_groups.iterrows()}
        label_pnfcnc = {row.num_clusters: 100 * (row.num_new_failure_classes - row.num_new_failure_classes_covered) / row.num_new_failure_classes if row.num_new_failure_classes > 0 else np.nan for index, row in label_groups.iterrows()}
        label_pfnc_upper = {}
        label_pnfnc_upper = {}
        label_pnfcnc_upper = {}
        sorted_label_num_clusters = sorted(list(label_pfnc.keys()))
        for num_clusters in sorted_label_num_clusters:
            label_pfnc_upper[num_clusters] = max(label_pfnc[num_clusters], max(label_pfnc_upper.values()) if len(label_pfnc_upper) > 0 else 0)
            label_pnfnc_upper[num_clusters] = max(label_pnfnc[num_clusters], max(label_pnfnc_upper.values()) if len(label_pnfnc_upper) > 0 else 0)
            label_pnfcnc_upper[num_clusters] = max(label_pnfcnc[num_clusters], max(label_pnfcnc_upper.values()) if len(label_pnfcnc_upper) > 0 else 0)
        for precision in [0]:
            # perc_failures_unseen_dict[f'Label bin {precision}'].append(1 - label_groups.loc[np.isclose(label_groups.precision, precision)]['seen_perc'].values[0])
            perc_tests_unseen_dict[f'Label bin {precision}'].append(1 - label_groups.loc[np.isclose(label_groups.precision, precision)]['seen_perc'].values[0])
            perc_failures_unseen_dict[f'Label bin {precision}'].append(1-label_groups.loc[np.isclose(label_groups.precision, precision)]['fail_perc'].values[0])  # THIS IS CORRECT
            # perc_failures_unseen_dict[f'Label bin {precision}'].append(label_groups.loc[np.isclose(label_groups.precision, precision)]['unseen_test_fails'].values[0] / 100)
            # perc_failures_unseen_dict[f'Label bin {precision}'].append(1e-2*(1-label_groups.loc[np.isclose(label_groups.precision, precision)]['fail_perc'].values[0]) / (1-label_groups.loc[np.isclose(label_groups.precision, precision)]['seen_perc'].values[0]) if (1-label_groups.loc[np.isclose(label_groups.precision, precision)]['seen_perc'].values[0]) > 0 else 0)
            pnfnc = label_groups.loc[np.isclose(label_groups.precision, precision)].apply(lambda x: (x.num_new_failures - x.num_new_failures_covered) / x.num_new_failures if x.num_new_failures > 0 else np.nan, axis=1)
            pnfnc_classes = label_groups.loc[np.isclose(label_groups.precision, precision)].apply(lambda x: (x.num_new_failure_classes - x.num_new_failure_classes_covered) / x.num_new_failure_classes if x.num_new_failure_classes > 0 else np.nan, axis=1)
            new_perc_failures_unseen_dict[f'Label bin {precision}'].append(pnfnc.values[0])
            new_perc_fail_classes_unseen_dict[f'Label bin {precision}'].append(pnfnc_classes.values[0])
            perc_consistent_class = label_groups.loc[np.isclose(label_groups.precision, precision)].apply(lambda x: x.num_consistent_non_singleton_classes / x.num_non_singleton_classes, axis=1)
            perc_possible_inconsistent_class = label_groups.loc[np.isclose(label_groups.precision, precision)].apply(lambda x: (x.num_non_singleton_classes - x.num_consistent_non_singleton_classes) / min(x.num_clusters, x.total_failures),axis=1)
            perc_consistent_class_dict[f'Label bin {precision}'].append(perc_consistent_class.values[0])
            perc_possible_inconsistent_class_dict[f'Label bin {precision}'].append(perc_possible_inconsistent_class.values[0])
            # perc_classes_with_failures_that_are_consistent = label_groups.loc[np.isclose(label_groups.precision, precision)].apply(lambda x: x.num_classes_with_all_failures / x.num_classes_with_failures, axis=1)
            perc_classes_with_failures_that_are_consistent = label_groups.loc[np.isclose(label_groups.precision, precision)].apply(lambda x: (x.num_classes_with_all_failures / x.num_classes_with_failures), axis=1)
            # print(f'Label bin {precision}')
            # print(perc_classes_with_failures_that_are_consistent.values[0])
            perc_classes_with_failures_that_are_consistent_class_dict[f'Label bin {precision}'].append(perc_classes_with_failures_that_are_consistent.values[0])
            num_cluster_dict[f'Label bin {precision}'].append(label_groups.loc[np.isclose(label_groups.precision, precision)]['num_clusters'].values[0])
        for random_file in glob.glob(f'{folder}carla_abstract/random_clusters_*.csv'):
            num_clusters = int(random_file[random_file.rfind('_')+1:-4])
            label_groups = pd.read_csv(random_file)
            random_num_cluster_dict[f'random_{num_clusters}'] = num_clusters
            random_perc_failures_unseen_dict[f'random_{num_clusters}'] = 1-label_groups.loc[label_groups['trial'] == -1]['fail_perc'].values[0]
        for random_file in glob.glob(f'{folder}carla_abstract/singleton_random_clusters_*.csv'):
            num_clusters = int(random_file[random_file.rfind('_') + 1:-4])
            label_groups = pd.read_csv(random_file)
            singleton_random_num_cluster_dict[f'random_{num_clusters}'] = num_clusters
            singleton_random_perc_failures_unseen_dict[f'random_{num_clusters}'] = 1 - label_groups.loc[label_groups['trial'] == -1]['fail_perc'].values[0]
        true_random = pd.read_csv(f'{folder}carla_abstract/true_random.csv')
        true_random_num_clusters = sorted(list(set(true_random['num_clusters'].tolist())))
        true_random_ptnc = {num_clusters: 1 - true_random[true_random['num_clusters'] == num_clusters]['fail_perc'].mean() for num_clusters in true_random_num_clusters}
        true_random_pfnc = {num_clusters: 1 - true_random[true_random['num_clusters'] == num_clusters]['fail_perc'].mean() for num_clusters in true_random_num_clusters}
        true_random['pnfnc'] = true_random.apply(lambda x: (x.num_new_failures - x.num_new_failures_covered) / x.num_new_failures if x.num_new_failures > 0 else np.nan, axis=1)
        true_random['pnfnc_classes'] = true_random.apply(lambda x: (x.num_new_failure_classes - x.num_new_failure_classes_covered) / x.num_new_failure_classes if x.num_new_failure_classes > 0 else np.nan,axis=1)
        true_random_new_perc_failures_unseen_dict = {num_clusters: true_random[true_random['num_clusters'] == num_clusters]['pnfnc'].mean() for num_clusters in true_random_num_clusters}
        true_random_new_perc_fail_classes_unseen_dict = {num_clusters: true_random[true_random['num_clusters'] == num_clusters]['pnfnc_classes'].mean() for num_clusters in true_random_num_clusters}
        true_random['perc_consistent_class'] = true_random.apply(lambda x: x.num_consistent_non_singleton_classes / x.num_non_singleton_classes, axis=1)
        true_random['perc_possible_inconsistent_class'] = true_random.apply(lambda x: (x.num_non_singleton_classes - x.num_consistent_non_singleton_classes) / min(x.num_clusters, x.total_failures),axis=1)
        true_random_perc_consistent_class_dict = {num_clusters: true_random[true_random['num_clusters'] == num_clusters]['perc_consistent_class'].mean() for num_clusters in true_random_num_clusters}
        true_random_perc_possible_inconsistent_class_dict = {num_clusters: true_random[true_random['num_clusters'] == num_clusters]['perc_possible_inconsistent_class'].mean() for num_clusters in true_random_num_clusters}
        true_random['perc_classes_with_failures_that_are_consistent'] = true_random.apply(lambda x: x.num_classes_with_all_failures / x.num_classes_with_failures, axis=1)
        true_random_perc_classes_with_failures_that_are_consistent_class_dict = {num_clusters: true_random[true_random['num_clusters'] == num_clusters]['perc_classes_with_failures_that_are_consistent'].mean() for num_clusters in true_random_num_clusters}
        # true_random.to_csv(f'{folder}carla_abstract/true_random_temp.csv')

    eighty_index = [index for index, val in enumerate(x) if abs(val - 0.8) < 1e-5][0]

    plt.rcParams['axes.prop_cycle'] = cycler('color', matplotlib.colormaps['tab20'].colors)

    fig = plt.figure()
    ax = fig.gca()
    for index, graph_type in enumerate(perc_failures_unseen_dict):
        ax.scatter(num_cluster_dict[graph_type], perc_failures_unseen_dict[graph_type], color=get_color(graph_type, index), label=f'{graph_type}')
    legend = ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax.set_xlabel('Number of clusters Train+Test (lower is better)')
    ax.set_ylabel('Fraction of Failures that are Unseen (higher is better)')
    fig.savefig(f'{output_path}/num_clusters.png', bbox_extra_artists=(legend,), bbox_inches='tight')
    fig.savefig(f'{output_path}/num_clusters.svg', bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    ax = fig.gca()
    for index, graph_type in enumerate(perc_failures_unseen_dict):
        ax.scatter(num_cluster_dict[graph_type][eighty_index], perc_failures_unseen_dict[graph_type][eighty_index], color=get_color(graph_type, index),
                   label=f'{graph_type}')
    legend = ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax.set_xlabel('Number of clusters Train+Test (lower is better)')
    ax.set_ylabel('Fraction of Failures that are Unseen (higher is better)')
    fig.savefig(f'{output_path}/num_clusters_80_20.png', bbox_extra_artists=(legend,), bbox_inches='tight')
    fig.savefig(f'{output_path}/num_clusters_80_20.svg', bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close(fig)

    plt.rcParams['axes.prop_cycle'] = cycler('color', matplotlib.colors.TABLEAU_COLORS)
    fig = plt.figure()
    ax = fig.gca()
    for index, graph_type in enumerate(graphs_to_show):
        label = label_map[graph_type] if graph_type in label_map else graph_type
        print(graph_type)
        ax.scatter(num_cluster_dict[graph_type][eighty_index],
                   100 * perc_failures_unseen_dict[graph_type][eighty_index],
                   color=get_color(graph_type, index),
                   label=f'{label}',
                   marker=get_marker(graph_type))
    ax.scatter(num_frames, 100, color='k', label='All Data Unique\n(Trivial Solution)', marker='*')
    legend = ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax.set_xlabel('Number of clusters Train+Test')
    ax.set_ylabel('Percentage of Failures that are not Covered')
    fig.savefig(f'{output_path}/num_clusters_80_20_trivial.png', bbox_extra_artists=(legend,), bbox_inches='tight')
    fig.savefig(f'{output_path}/num_clusters_80_20_trivial.svg', bbox_extra_artists=(legend,), bbox_inches='tight')
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
                   color=get_color(graph_type, index),
                   label=f'{label}',
                   marker=get_marker(graph_type))
    ax.scatter(num_frames, 100, color='k', label='All Data Unique\n(Trivial Solution)', marker='*')
    ax.hlines(20, 0, num_frames, linestyles='dashed', color='k', label='Random Baseline')
    ax.plot([0, num_frames], [0, 100], color='k', label='_nolabel_')
    # ax.hlines(1, 0, num_frames, linestyles='dashed', color='k', label='Even')
    legend = ax.legend()
    ax.set_xlabel('Total Number of Equivalence Classes')
    ax.set_ylabel('Percentage of Failures Not Covered (PFNC)')
    # ax.set_ylabel('% fail unseen / % test unseen')
    fig.savefig(f'{output_path}/num_clusters_80_20_trivial_inner_legend.png', bbox_extra_artists=(legend,), bbox_inches='tight')
    fig.savefig(f'{output_path}/num_clusters_80_20_trivial_inner_legend.svg', bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    ax = fig.gca()
    for index, graph_type in enumerate(graphs_to_show):
        label = label_map[graph_type] if graph_type in label_map else graph_type
        ax.scatter(num_cluster_dict[graph_type][eighty_index],
                   100 * perc_failures_unseen_dict[graph_type][eighty_index],
                   color=get_color(graph_type, index),
                   label=f'{label}',
                   marker=get_marker(graph_type))
    for nc in true_random_num_clusters:
        ax.scatter(nc,
                   100 * (true_random_pfnc[nc]),
                   color=f'r',
                   label=f'_nolabel_',
                   marker='4')
    ax.scatter(nc,
               100 * (true_random_pfnc[nc]),
               color=f'r',
               label=f'True Random',
               marker='4')
    N = np.linspace(1, 1000000, num=10000)
    M = num_frames
    T = 36809
    theory_y = 100 * np.power((N - 1) / N, T)
    theory_x = N - N * np.power((N - 1) / N, M)
    ax.plot(theory_x, theory_y, color='k', ls='--', label='True Random')
    sorted_label_num_clusters = sorted(list(label_pfnc.keys()))
    # ax.plot(sorted_label_num_clusters, [label_pfnc[x] for x in sorted_label_num_clusters], color='b', ls='--', label='GT Steering Upper Bound')
    ax.scatter(sorted_label_num_clusters, [label_pfnc[x] for x in sorted_label_num_clusters], color='b', marker='*', label='GT Steering Upper Bound')
    ax.scatter(num_frames, 100, color='k', label='All Data Unique\n(Trivial Solution)', marker='*')
    legend = ax.legend(bbox_to_anchor=(0, -.1), loc='upper left', ncols=2)
    # ax.set_xlabel('Number of clusters Train+Test (lower is better)')
    # ax.set_ylabel('Percentage of Failures that are not Covered (higher is better)')
    ax.set_xlabel('Number of clusters Train+Test')
    ax.set_ylabel('Percentage of Failures that are not Covered')
    fig.savefig(f'{output_path}/num_clusters_80_20_trivial_legend_below.png', bbox_extra_artists=(legend,),
                bbox_inches='tight')
    fig.savefig(f'{output_path}/num_clusters_80_20_trivial_legend_below.svg', bbox_extra_artists=(legend,),
                bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    ax = fig.gca()
    for index, graph_type in enumerate(graphs_to_show):
        label = label_map[graph_type] if graph_type in label_map else graph_type
        ax.scatter(num_cluster_dict[graph_type][eighty_index],
                   100 * perc_tests_unseen_dict[graph_type][eighty_index],
                   color=get_color(graph_type, index),
                   label=f'{label}',
                   marker=get_marker(graph_type))
    for nc in true_random_num_clusters:
        ax.scatter(nc,
                   100 * (true_random_ptnc[nc]),
                   color=f'r',
                   label=f'_nolabel_',
                   marker='4')
    ax.scatter(nc,
               100 * (true_random_ptnc[nc]),
               color=f'r',
               label=f'True Random',
               marker='4')
    N = np.linspace(1, 1000000, num=10000)
    M = num_frames
    T = 36809
    theory_y = 100 * np.power((N - 1) / N, T)
    theory_x = N - N * np.power((N - 1) / N, M)
    ax.plot(theory_x, theory_y, color='k', ls='--', label='True Random')
    ax.scatter(num_frames, 100, color='k', label='All Data Unique\n(Trivial Solution)', marker='*')
    legend = ax.legend(bbox_to_anchor=(0, -.1), loc='upper left', ncols=2)
    ax.set_xlabel('Number of clusters Train+Test')
    ax.set_ylabel('Percentage of Tests that are not Covered')
    fig.savefig(f'{output_path}/ptnc.png', bbox_extra_artists=(legend,),
                bbox_inches='tight')
    fig.savefig(f'{output_path}/ptnc.svg', bbox_extra_artists=(legend,),
                bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    ax = fig.gca()
    print('PNFNC')
    for index, graph_type in enumerate(graphs_to_show):
        label = label_map[graph_type] if graph_type in label_map else graph_type
        if graph_type not in new_perc_failures_unseen_dict:
            continue
        print(f'{label} num_clusters:', num_cluster_dict[graph_type][eighty_index])
        print(f'{label}:', 100 * new_perc_failures_unseen_dict[graph_type][eighty_index])
        ax.scatter(num_cluster_dict[graph_type][eighty_index],
                   100 * new_perc_failures_unseen_dict[graph_type][eighty_index],
                   color=get_color(graph_type, index),
                   label=f'{label}',
                   marker=get_marker(graph_type))
    # for singleton_random_num_clusters in singleton_random_num_cluster_dict:
    #     ax.scatter(singleton_random_num_cluster_dict[singleton_random_num_clusters],
    #                100 * singleton_random_perc_failures_unseen_dict[singleton_random_num_clusters],
    #                color=f'b',
    #                label=f'_nolabel_',
    #                marker='1')
    # one_key = list(singleton_random_num_cluster_dict.keys())[0]
    # ax.scatter(singleton_random_num_cluster_dict[one_key],
    #            100 * singleton_random_perc_failures_unseen_dict[one_key],
    #            color=f'b',
    #            label=f'Random Max Singletons',
    #            marker='1')
    x, y = normalize_random(true_random_num_clusters, [100 * true_random_new_perc_failures_unseen_dict[nc] for nc in true_random_num_clusters])
    # ax.plot(x, y, color='k', ls='--', label='Random Clustering')
    ax.scatter(x, y, color='k', marker='4', label='Random Clustering')
    ax.scatter(num_frames, 100, color='k', label='All Data Unique\n(Trivial Solution)', marker='*')
    ax.set_xlabel('Number of Equivalence Classes Train+Test')
    ax.set_ylabel('Percentage of Novel Failures Not Covered (PNFNC)')
    legend = ax.legend(loc='lower right')
    fig.savefig(f'{output_path}/new_num_clusters_80_20_trivial_legend_within.png', bbox_extra_artists=(legend,),
                bbox_inches='tight')
    fig.savefig(f'{output_path}/new_num_clusters_80_20_trivial_legend_within.svg', bbox_extra_artists=(legend,),
                bbox_inches='tight')
    legend = ax.legend(bbox_to_anchor=(0.5, -.1), loc='upper center', ncols=2)
    fig.savefig(f'{output_path}/new_num_clusters_80_20_trivial_legend_below.png', bbox_extra_artists=(legend,),
                bbox_inches='tight')
    fig.savefig(f'{output_path}/new_num_clusters_80_20_trivial_legend_below.svg', bbox_extra_artists=(legend,),
                bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    ax = fig.gca()
    for index, graph_type in enumerate(graphs_to_show):
        label = label_map[graph_type] if graph_type in label_map else graph_type
        if graph_type not in new_perc_fail_classes_unseen_dict:
            continue
        ax.scatter(num_cluster_dict[graph_type][eighty_index],
                   100 * new_perc_fail_classes_unseen_dict[graph_type][eighty_index],
                   color=get_color(graph_type, index),
                   label=f'{label}',
                   marker=get_marker(graph_type))
    # for singleton_random_num_clusters in singleton_random_num_cluster_dict:
    #     ax.scatter(singleton_random_num_cluster_dict[singleton_random_num_clusters],
    #                100 * singleton_random_perc_failures_unseen_dict[singleton_random_num_clusters],
    #                color=f'b',
    #                label=f'_nolabel_',
    #                marker='1')
    # one_key = list(singleton_random_num_cluster_dict.keys())[0]
    # ax.scatter(singleton_random_num_cluster_dict[one_key],
    #            100 * singleton_random_perc_failures_unseen_dict[one_key],
    #            color=f'b',
    #            label=f'Random Max Singletons',
    #            marker='1')
    x, y = normalize_random(true_random_num_clusters,
                            [100 * true_random_new_perc_fail_classes_unseen_dict[nc] for nc in true_random_num_clusters])
    # ax.plot(x, y, color='k', ls='--', label='Random Clustering')
    ax.scatter(x, y, color='k', marker='4', label='Random Clustering')
    # for nc in true_random_num_clusters:
    #     ax.scatter(nc,
    #                100 * (true_random_new_perc_fail_classes_unseen_dict[nc]),
    #                color=f'r',
    #                label=f'_nolabel_',
    #                marker='4')
    # ax.scatter(nc,
    #            100 * (true_random_new_perc_fail_classes_unseen_dict[nc]),
    #            color=f'r',
    #            label=f'True Random',
    #            marker='4')
    # N = np.linspace(1, 1000000, num=10000)
    # M = num_frames
    # T = 36809
    # theory_y = 100 * np.power((N - 1) / N, T)
    # theory_x = N - N * np.power((N - 1) / N, M)
    # ax.plot(theory_x, theory_y, color='k', ls='--', label='True Random')
    # # ax.hlines(20, 0, num_frames, linestyles='dashed', color='k', label='Random Baseline')
    # # ax.plot([0, num_frames], [0, 100], color='k', label='_nolabel_')
    # for random_num_clusters in random_num_cluster_dict:
    #     ax.scatter(random_num_cluster_dict[random_num_clusters],
    #                100 * random_perc_failures_unseen_dict[random_num_clusters],
    #                color=f'k',
    #                label=f'_nolabel_',
    #                marker='2')
    # one_key = list(random_num_cluster_dict.keys())[0]
    # ax.scatter(random_num_cluster_dict[one_key],
    #            100 * random_perc_failures_unseen_dict[one_key],
    #            color=f'k',
    #            label=f'Uniform Random Baseline',
    #            marker='2')
    ax.scatter(num_frames, 100, color='k', label='All Data Unique\n(Trivial Solution)', marker='*')
    # sorted_label_num_clusters = sorted(list(label_pnfcnc.keys()))
    # ax.plot(sorted_label_num_clusters, [label_pnfcnc_upper[x] for x in sorted_label_num_clusters], color='b', ls='--',label='GT Steering Upper Bound')
    legend = ax.legend(bbox_to_anchor=(0.5, -.1), loc='upper center', ncols=2)
    # ax.set_xlabel('Number of clusters Train+Test (lower is better)')
    # ax.set_ylabel('Percentage of Failures that are not Covered (higher is better)')
    ax.set_xlabel('Number of Equivalence Classes Train+Test')
    ax.set_ylabel('Percentage of Novel Failure Classes that are not Covered')
    fig.savefig(f'{output_path}/new_classes_num_clusters_80_20_trivial_legend_below.png', bbox_extra_artists=(legend,),
                bbox_inches='tight')
    fig.savefig(f'{output_path}/new_classes_num_clusters_80_20_trivial_legend_below.svg', bbox_extra_artists=(legend,),
                bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    ax = fig.gca()
    for index, graph_type in enumerate(graphs_to_show):
        label = label_map[graph_type] if graph_type in label_map else graph_type
        if graph_type not in perc_consistent_class_dict:
            continue
        ax.scatter(num_cluster_dict[graph_type][eighty_index],
                   100 * perc_consistent_class_dict[graph_type][eighty_index],
                   color=get_color(graph_type, index),
                   label=f'{label}',
                   marker=get_marker(graph_type))
    for nc in true_random_num_clusters:
        ax.scatter(nc,
                   100 * (true_random_perc_consistent_class_dict[nc]),
                   color=f'r',
                   label=f'_nolabel_',
                   marker='4')
    ax.scatter(nc,
               100 * (true_random_perc_consistent_class_dict[nc]),
               color=f'r',
               label=f'True Random',
               marker='4')
    legend = ax.legend(bbox_to_anchor=(0, -.1), loc='upper left', ncols=2)
    ax.set_xlabel('Number of clusters Train+Test')
    ax.set_ylabel('% of Non-Singleton Classes Consistent (higher better)')
    fig.savefig(f'{output_path}/perc_non_singleton_consistent_legend_below.png', bbox_extra_artists=(legend,),
                bbox_inches='tight')
    fig.savefig(f'{output_path}/perc_non_singleton_consistent_legend_below.svg', bbox_extra_artists=(legend,),
                bbox_inches='tight')
    plt.close(fig)

    # fig = plt.figure()
    # ax = fig.gca()
    # for index, graph_type in enumerate(graphs_to_show):
    #     label = label_map[graph_type] if graph_type in label_map else graph_type
    #     if graph_type not in perc_consistent_class_dict:
    #         continue
    #     ax.scatter(num_cluster_dict[graph_type][eighty_index],
    #                np.log(perc_consistent_class_dict[graph_type][eighty_index]),
    #                color=get_color(graph_type, index),
    #                label=f'{label}',
    #                marker=get_marker(graph_type))
    # for nc in true_random_num_clusters:
    #     ax.scatter(nc,
    #                np.log(true_random_perc_consistent_class_dict[nc]),
    #                color=f'r',
    #                label=f'_nolabel_',
    #                marker='4')
    # ax.scatter(nc,
    #            np.log(true_random_perc_consistent_class_dict[nc]),
    #            color=f'r',
    #            label=f'True Random',
    #            marker='4')
    # legend = ax.legend(bbox_to_anchor=(0, -.1), loc='upper left', ncols=2)
    # ax.set_xlabel('Number of clusters Train+Test')
    # ax.set_ylabel('LOG % of Non-Singleton Classes Consistent (higher better)')
    # fig.savefig(f'{output_path}/log_perc_non_singleton_consistent_legend_below.png', bbox_extra_artists=(legend,),
    #             bbox_inches='tight')
    # fig.savefig(f'{output_path}/log_perc_non_singleton_consistent_legend_below.svg', bbox_extra_artists=(legend,),
    #             bbox_inches='tight')
    # plt.close(fig)

    fig = plt.figure()
    ax = fig.gca()
    for index, graph_type in enumerate(graphs_to_show):
        label = label_map[graph_type] if graph_type in label_map else graph_type
        if graph_type not in perc_possible_inconsistent_class_dict:
            continue
        ax.scatter(num_cluster_dict[graph_type][eighty_index],
                   100 * perc_possible_inconsistent_class_dict[graph_type][eighty_index],
                   color=get_color(graph_type, index),
                   label=f'{label}',
                   marker=get_marker(graph_type))
    for nc in true_random_num_clusters:
        ax.scatter(nc,
                   100 * (true_random_perc_possible_inconsistent_class_dict[nc]),
                   color=f'r',
                   label=f'_nolabel_',
                   marker='4')
    ax.scatter(nc,
               100 * (true_random_perc_possible_inconsistent_class_dict[nc]),
               color=f'r',
               label=f'True Random',
               marker='4')
    legend = ax.legend(bbox_to_anchor=(0, -.1), loc='upper left', ncols=2)
    ax.set_xlabel('Number of clusters Train+Test')
    ax.set_ylabel('Percentage of Possible Inconsistent Classes (lower better)')
    fig.savefig(f'{output_path}/perc_possible_inconsistent_legend_below.png', bbox_extra_artists=(legend,),
                bbox_inches='tight')
    fig.savefig(f'{output_path}/perc_possible_inconsistent_legend_below.svg', bbox_extra_artists=(legend,),
                bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    ax = fig.gca()
    for index, graph_type in enumerate(graphs_to_show):
        label = label_map[graph_type] if graph_type in label_map else graph_type
        if graph_type not in perc_classes_with_failures_that_are_consistent_class_dict:
            continue
        ax.scatter(num_cluster_dict[graph_type][eighty_index],
                   100 * perc_classes_with_failures_that_are_consistent_class_dict[graph_type][eighty_index],
                   color=get_color(graph_type, index),
                   label=f'{label}',
                   marker=get_marker(graph_type))
    for nc in true_random_num_clusters:
        ax.scatter(nc,
                   100 * (true_random_perc_classes_with_failures_that_are_consistent_class_dict[nc]),
                   color=f'r',
                   label=f'_nolabel_',
                   marker='4')
    ax.scatter(nc,
               100 * (true_random_perc_classes_with_failures_that_are_consistent_class_dict[nc]),
               color=f'r',
               label=f'True Random',
               marker='4')
    legend = ax.legend(bbox_to_anchor=(0, -.1), loc='upper left', ncols=2)
    ax.set_xlabel('Number of clusters Train+Test')
    ax.set_ylabel('% of classes with failures that are consistent (higher is better)')
    fig.savefig(f'{output_path}/perc_failure_classes_consistent.png', bbox_extra_artists=(legend,),
                bbox_inches='tight')
    fig.savefig(f'{output_path}/perc_failure_classes_consistent.svg', bbox_extra_artists=(legend,),
                bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    meta_figure(sys.argv[1:])
