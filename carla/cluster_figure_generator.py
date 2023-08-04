import argparse
import copy
import os
import sys

from utils.dataset import Dataset
from pathlib import Path
import matplotlib.pyplot as plt
import glob

from meta_figure_generator import label_map, graphs_to_show

def custom_argparse(arg_string):
    parser = argparse.ArgumentParser(
        description="Validation Split Visualizer"
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


def cluster_figure_generator(arg_string):
    args = custom_argparse(arg_string)
    output_path = args.output_path/''
    os.makedirs(output_path, exist_ok=True)
    datasets = {}
    x_vals = {}
    box_vals = {}
    for dataset_file in glob.glob(str(args.input_path/'*.json')):
        graph_type = dataset_file[dataset_file.rfind('/') + 1:dataset_file.rfind('.')]
        if graph_type not in graphs_to_show:
            continue
        datasets[graph_type] = Dataset.load_from_file(dataset_file, '', '')
        cumulative = []
        count = 0
        box_vals[graph_type] = []
        for cluster_key in datasets[graph_type]._sorted_cluster_keys:
            cluster = datasets[graph_type]._clusters[cluster_key]
            count += len(cluster)
            cumulative.append(count)
            box_vals[graph_type].append(len(cluster))
        x_vals[graph_type] = cumulative
    orig_x_vals = copy.deepcopy(x_vals)
    max_index = max([len(x) for x in x_vals.values()])
    max_frames = max([x[-1] for x in x_vals.values()])

    for index, graph_type in enumerate(graphs_to_show):
        if 'Label' in graph_type:
            continue
        # fig = plt.figure(figsize=(6.4 * 3, 4.8 * 2))
        fig = plt.figure()
        # fig = plt.figure(figsize=(6.4 * 2, 4.8))
        ax1 = fig.gca()
        ax2 = ax1.twinx()
        vals = box_vals[graph_type]
        color = 'tab:blue'
        # ax1.bar([i for i in range(len(vals))], vals, width=1, color=color)
        scatter = ax1.scatter([i for i in range(len(vals))], vals, color=color, label='Images in Class (left)')
        # ax1.text(0, vals[0], f'Largest Class Covers {vals[0]} Images ({100*vals[0] / max_frames:.2f}%)')
        ax1.set_xlabel('Equivalence Class ID')
        ax1.set_ylabel('Number of Images in Class', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        color = 'tab:red'
        vals = x_vals[graph_type]
        cur_max_index = len(vals)
        x_text_offset = cur_max_index / 100
        y_text_offset = vals[-1] / 40
        line, = ax2.plot([i for i in range(len(vals))], vals, color=color, label='Cumulative Images Covered (right)')
        lines = [scatter, line]
        ax1.legend(lines, [line.get_label() for line in lines], loc='lower right', bbox_to_anchor=(1, 0.05))
        singleton_index = min([index for index, val in enumerate(box_vals[graph_type]) if val == 1]) - 1
        ax2.text(singleton_index + x_text_offset, vals[singleton_index]+y_text_offset, f'Remaining {cur_max_index - singleton_index} Images in Singleton Classes\n({100 * (max_frames - vals[singleton_index]) / max_frames:.2f}% of Images in {100*(cur_max_index - singleton_index)/(cur_max_index):.2f}% of Classes)')
        ax2.hlines(vals[singleton_index], singleton_index, cur_max_index, label='80% of Images Covered', color='k')
        # eighty_index = min([index for index, val in enumerate(vals) if val >= max_frames * 0.8])
        # ax2.text(eighty_index + x_text_offset, vals[eighty_index]+y_text_offset, f'80% of Images Covered by Largest {eighty_index+1} Classes\n({100 * vals[eighty_index] / max_frames:.2f}% of Images in {100*(eighty_index+1)/(cur_max_index):.2f}% of Classes)')
        # # ax2.vlines(eighty_index, 0, vals[eighty_index], label='80% of Images Covered', color='k')
        # ax2.hlines(vals[eighty_index], eighty_index, cur_max_index, label='80% of Images Covered', color='k')
        ax2.text(10+x_text_offset, vals[10]+y_text_offset, f'Largest 10 Classes Cover {vals[10]} Images\n({100 * vals[10] / max_frames:.2f}% of Images in {100*10/cur_max_index:.2f}% of Classes)')
        ax2.hlines(vals[10], 10, cur_max_index, label='Largest 10 Classes', color='k')
        ax2.text(0+x_text_offset, vals[0]+y_text_offset, f'Largest Class Covers {vals[0]} Images\n({100*vals[0] / max_frames:.2f}% of Images in {100/cur_max_index:.2f}% of Classes)')
        ax2.hlines(vals[0], 0, cur_max_index, label='Largest Class', color='k')
        ax2.set_ylim(bottom=0)
        ax1.set_ylim(bottom=0)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_xlabel('Equivalence Class ID')
        ax2.set_ylabel('Cumulative Images Covered', color=color)
        fig.suptitle(f'{label_map[graph_type]} Equivalence Class Partitions')  # , fontsize=16
        fig.savefig(f'{output_path}/cluster_viz_{graph_type}.png', bbox_inches='tight')
        plt.close(fig)

    fig = plt.figure()
    ax = fig.gca()
    for index, graph_type in enumerate(graphs_to_show):
        if 'Label' in graph_type:
            continue
        vals = x_vals[graph_type]
        ax.plot([i for i in range(len(vals))], vals, color=f'C{index}', label=f'{label_map[graph_type]}')
    # ax.plot(x, x, linestyle='-', color='k', label='All Data Unique')
    legend = ax.legend()
    ax.set_xlabel('Number of Equivalence Classes')
    ax.set_ylabel('Number of Images Covered')
    fig.savefig(f'{output_path}/cluster_viz.png', bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close(fig)

    for graph_type, x in x_vals.items():
        if len(x) < max_index:
            x.extend([x[-1]] * (max_index - len(x)))
    
    fig = plt.figure()
    ax = fig.gca()
    for index, graph_type in enumerate(graphs_to_show):
        if 'Label' in graph_type:
            continue
        vals = x_vals[graph_type]
        ax.plot([i for i in range(len(vals))], vals, color=f'C{index}', label=f'{label_map[graph_type]}')
    # ax.plot(x, x, linestyle='-', color='k', label='All Data Unique')
    legend = ax.legend()
    ax.set_xlabel('Number of Equivalence Classes')
    ax.set_ylabel('Number of Images Covered')
    fig.savefig(f'{output_path}/cluster_viz_max_fill.png', bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    ax = fig.gca()
    lower_xlim = -2000
    ax.set_xlim(left=lower_xlim, right=max_index-lower_xlim)
    extra_x_ticks = set()
    extra_y_ticks = {0}
    for index, graph_type in enumerate(graphs_to_show):
        if 'Label' in graph_type:
            continue
        vals = x_vals[graph_type]
        ax.plot([i for i in range(len(vals))], vals, color=f'C{index}', label=f'{label_map[graph_type]}')
        # print(graph_type, len(orig_x_vals[graph_type]))
        print(graph_type, orig_x_vals[graph_type][0])
        if index in [0,1]:
            ls = (0, (5, 7))
        elif index == 4:
            ls = (0, (5, 1))
        else:
            ls = (6, (5, 7))
        # ls = (0, (5, 7)) if index in [0,1,4] else (6, (5, 7))
        ax.vlines(len(orig_x_vals[graph_type]), 0, orig_x_vals[graph_type][-1], color=f'C{index}', linestyle='--', label='_none')
        ax.hlines(orig_x_vals[graph_type][0], lower_xlim, 0, color=f'C{index}', linestyle=ls, label='_none')
        extra_x_ticks.add(len(orig_x_vals[graph_type]))
        extra_y_ticks.add(orig_x_vals[graph_type][0])
        # ax.text(lower_xlim, orig_x_vals[graph_type][0], str(orig_x_vals[graph_type][0]))
    eighty_percent = int(round(0.8*max_frames))
    ax.hlines(eighty_percent, lower_xlim, max_index, color='k', label='80% of Images')
    extra_y_ticks.add(eighty_percent)
    # ax.plot(x, x, linestyle='-', color='k', label='All Data Unique')
    extra_y_ticks.add(max_frames)
    legend = ax.legend()
    # adapted from https://stackoverflow.com/questions/14716660/adding-extra-axis-ticks-using-matplotlib
    ax.set_xticks(list(extra_x_ticks))
    ax.set_yticks(list(extra_y_ticks))
    ax.set_xticklabels(ax.get_xticks(), rotation=75)
    # ax.set_yticklabels(ax.get_yticks(), rotation=75)
    ax.set_xlim(left=lower_xlim, right=max_index-lower_xlim)
    ax.set_xlabel('Number of Equivalence Classes')
    ax.set_ylabel('Number of Images Covered')
    # fig.autofmt_xdate()
    fig.savefig(f'{output_path}/cluster_viz_max_fill_lines.png', bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    cluster_figure_generator(sys.argv[1:])