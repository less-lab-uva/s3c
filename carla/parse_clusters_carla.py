import argparse
import copy
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

import cv2
import scipy
from tqdm import tqdm

import venn

from utils.dataset import Dataset
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def custom_argparse(arg_string):
    parser = argparse.ArgumentParser(
        description="Validation Split Visualizer"
    )
    parser.add_argument(
        "-dsf",
        "--dataset_file",
        type=Path,
        required=True,
        help="Location to load data set file."
    )
    parser.add_argument(
        "-input_path",
        "--input_path",
        type=Path,
        required=False,
        help="Location of input images"
    )
    parser.add_argument(
        "-test",
        "--test_results",
        type=Path,
        nargs='+',
        required=True,
        help="Results of model on tests."
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=Path,
        required=True,
        help="Directory to save figures"
    )
    parser.add_argument(
        "-f",
        "--failure_thresh_deg",
        type=float,
        required=False,
        default=5.0,
        help="Failure threshold. Data ranges -1 to 1, default is 5/140 to approximate 5 deg"
    )
    parser.add_argument(
        "-filter",
        "--filter_deg",
        type=float,
        required=False,
        default=-1,
        help="If supplied, only run statistics on data that has a ground truth label <= this amount."
    )
    parser.add_argument(
        "-deg_factor",
        "--deg_factor",
        type=float,
        required=False,
        default=70.0,
        help="Scaling factor required to convert outputs to degrees. For CARLA, this is 70, which is the default."
    )
    parser.add_argument(
        "-save_images",
        "--save_images",
        action='store_true',
        help="If set, save the seen/unseen failure splits"
    )
    parser.add_argument(
        "-save_all",
        "--save_all",
        action='store_true',
        help="If set, save all images failure splits"
    )
    return parser.parse_args(arg_string)


def get_frame_split(df, image_file):
    row = df.loc[df["image_file"] == image_file]
    if len(row) != 1:
        raise ValueError()
    return row.split.tolist()[0]


# def cluster_size_correlation(dataset, fail_df, file, failure_thresh_deg, savedir=None):
#     fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,10))
#     for

def print_file(string, file):
    print(string)
    if file is not None:
        with open(file, 'a') as f:
            f.write(f'{string}\n')


def plot_failed_data(fail_df, file, failure_thresh_deg, savedir=None, fail_only=False, filter_deg=-1, log_file=None):
    cutoff_df = pd.DataFrame({
        'name': pd.Series(dtype='str'),
        'cluster_id': pd.Series(dtype='int'),
        'fail_count': pd.Series(dtype='int'),
        'fail_perc': pd.Series(dtype='float')
    })
    x = [i for i in range(len(fail_df))]
    # https://matplotlib.org/3.4.3/gallery/ticks_and_spines/multiple_yaxis_with_spines.html
    fig, ((train1, train2), (train3, train4)) = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
    test_fails = train1.twinx()
    test_fail_rate = train2.twinx()
    test_fails_log = train3.twinx()
    cumulative_failures = train4.twinx()
    num_test_fail = fail_df['num_test_fail'].to_numpy()
    num_train = fail_df['num_train'].to_numpy()
    cumulative_failures_y = []
    cumulative_seen = []
    total_fail = 0
    total_train = 0
    for fail, train in zip(num_test_fail, num_train):
        total_fail += fail
        total_train += train
        cumulative_failures_y.append(total_fail)
        cumulative_seen.append(total_train)
    for train in [train1, train2, train4]:
        train.plot(x, [i if i > 0 else np.nan for i in fail_df['num_train']],
                   label='Number seen in training', color='b')
        train.set_ylabel('Number of times seen in training')
    train3.plot(x, fail_df['num_train'].to_numpy() + 1, label='Number seen in training', color='b')
    train3.set_ylabel('Number of times seen in training (log)')
    test_fails.bar(x, fail_df['num_test_fail'], label='Number of Test Failures', color='r')
    test_fails.set_ylabel('Number of test failures')
    test_fails_log.bar(x, fail_df['num_test_fail'].to_numpy() + 1, label='Number of Test Failures', color='r')
    test_fails_log.set_ylabel('Number of test failures (log)')
    test_fail_rate.scatter(x, fail_df['test_fail_perc'], label='Test Fail %', color='r')
    test_fail_rate.set_ylabel('Failure rate in test set')
    cumulative_failures.plot(x, cumulative_failures_y, label='Cumulative failures', color='r')
    try:
        one_percent_point = min([(cluster_id, y) for cluster_id, y, seen in
                               zip(x, cumulative_failures_y, cumulative_seen) if num_train[cluster_id] < total_train * 0.001])
    except ValueError:
        one_percent_point = (x[-1], total_fail)
    cumulative_failures.hlines(one_percent_point[1], one_percent_point[0], x[-1], color='tab:purple', label='Each cluster < 0.1%')
    cumulative_failures.vlines(one_percent_point[0], 0, one_percent_point[1], color='tab:purple')
    cumulative_failures.text(one_percent_point[0], one_percent_point[1],
                             f'<0.1%: {one_percent_point[1]}/{total_fail} ({100*one_percent_point[1]/total_fail:.1f}%)')
    cutoff_df.loc[len(cutoff_df.index)] = ['0.1%', one_percent_point[0], one_percent_point[1], one_percent_point[1]/total_fail]
    print_file(f'Clusters of size < 0.1% of training data: {one_percent_point[1]}/{total_fail} ({100*one_percent_point[1]/total_fail:.1f}%)', log_file)
    try:
        eight_percent_point = max([(cluster_id, y) for cluster_id, y, seen in
                               zip(x, cumulative_failures_y, cumulative_seen) if seen < total_train * 0.8])
    except ValueError:
        eight_percent_point = (x[-1], total_fail)
    cumulative_failures.hlines(eight_percent_point[1], eight_percent_point[0], x[-1], color='m', label='80% of training data')
    cumulative_failures.vlines(eight_percent_point[0], 0, eight_percent_point[1], color='m')
    cumulative_failures.text(eight_percent_point[0], eight_percent_point[1],
                             f'80%: {eight_percent_point[1]}/{total_fail} ({100*eight_percent_point[1]/total_fail:.1f}%)')
    cutoff_df.loc[len(cutoff_df.index)] = ['80%', eight_percent_point[0], eight_percent_point[1],
                                           eight_percent_point[1] / total_fail]
    print_file(f'80% of training data: {eight_percent_point[1]}/{total_fail} ({100*eight_percent_point[1]/total_fail:.1f}%)', log_file)
    try:
        seen_once_point = max([(cluster_id, y) for cluster_id, y, train in
                            zip(x, cumulative_failures_y, fail_df['num_train']) if train > 1])
    except ValueError:
        seen_once_point = (x[-1], total_fail)
    cumulative_failures.hlines(seen_once_point[1], seen_once_point[0], x[-1], color='g', label='Seen once in training')
    cumulative_failures.vlines(seen_once_point[0], 0, seen_once_point[1], color='g')
    cumulative_failures.text(seen_once_point[0], seen_once_point[1],
                             f'Once: {seen_once_point[1]}/{total_fail} ({100 * seen_once_point[1] / total_fail:.1f}%)')
    cutoff_df.loc[len(cutoff_df.index)] = ['seen_once', seen_once_point[0], seen_once_point[1],
                                           seen_once_point[1] / total_fail]
    print_file(f'Seen once in training: {seen_once_point[1]}/{total_fail} ({100 * seen_once_point[1] / total_fail:.1f}%)', log_file)
    try:
        unseen_point = max([(cluster_id, y) for cluster_id, y, train in
                            zip(x, cumulative_failures_y, fail_df['num_train']) if train > 0])
    except ValueError:
        unseen_point = (x[-1], total_fail)
    cumulative_failures.hlines(unseen_point[1], unseen_point[0], x[-1], color='k', label='End of training')
    cumulative_failures.vlines(unseen_point[0], 0, unseen_point[1], color='k')
    cumulative_failures.text(unseen_point[0], unseen_point[1],
                             f'End: {unseen_point[1]}/{total_fail} ({100 * unseen_point[1] / total_fail:.1f}%)')
    cutoff_df.loc[len(cutoff_df.index)] = ['unseen', unseen_point[0], unseen_point[1],
                                           unseen_point[1] / total_fail]
    print_file(f'Seen in training: {unseen_point[1]}/{total_fail} ({100 * unseen_point[1] / total_fail:.1f}%)', log_file)
    # quarter_points = []
    # for i in [0.25, 0.5, 0.75]:
    #     quarter_points.append(min([(cluster_id, y) for cluster_id, y in zip(x, cumulative_failures_y) if y > total * i]))
    # for (a, b) in quarter_points:
    #     cumulative_failures.hlines(b, 0, a)
    #     cumulative_failures.vlines(a, 0, b)
    cumulative_failures.set_ylabel('Cumulative number of test failures')
    for ax in [train1, train2, test_fails, train4, cumulative_failures]:
        ax.set_ylim(bottom=0)
    test_fail_rate.set_ylim(bottom=0, top=1.1)
    train3.set_yscale('log')
    test_fails_log.set_yscale('log')
    train3.set_ylim(bottom=1)
    test_fails_log.set_ylim(bottom=1)
    for ax in [test_fails, test_fail_rate, test_fails_log, cumulative_failures]:
        # ax.legend(loc='upper right')
        ax.legend(loc=1)
    for train in [train1, train2, train3, train4]:
        train.set_xlabel('Cluster ID')
        # ax.legend(loc='upper center')
        # ax.legend(loc=0)
    train1.set_title('Training clusters compared to test failures')
    train2.set_title('Training clusters compared to test failure rate')
    train3.set_title('Training clusters compared to test failures (log)')
    train4.set_title('Training clusters compared to cumulative failure count')
    title = f'Data: {file}, Fail thresh: {failure_thresh_deg}, Fail only: {fail_only}'
    fig.suptitle(title)
    cutoff_df.loc[len(cutoff_df.index)] = ['total', x[-1], total_fail,
                                           total_fail / total_fail]
    if savedir is not None:
        filename = savedir/f'{file}_fail_{failure_thresh_deg}{"filter_deg_%0.2f" % filter_deg if filter_deg > 0 else ""}{"_fail_only" if fail_only else ""}.png'
        fig.savefig(filename)
        csv_filename = savedir / f'{file}_fail_{failure_thresh_deg}{"filter_deg_%0.2f" % filter_deg if filter_deg > 0 else ""}{"_fail_only" if fail_only else ""}.csv'
        cutoff_df.to_csv(csv_filename)
    else:
        plt.show()
        plt.close()

def compute_best_fit(x, y):
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    lof, stats = np.polynomial.Polynomial.fit(x, y, deg=1, full=True)
    unique_x = np.unique(x)
    plt.plot(unique_x, lof(unique_x), color='red')
    plt.plot(unique_x, [intercept + slope * xi for xi in unique_x])

def compute_true_random(num_clusters, train_data, test_data, is_in_training, trial, fail_count, len_test, random_seed, index):
    gen = np.random.default_rng(seed=random_seed+index)
    # int_info = np.iinfo(np.int32)
    # # must be non-negative
    # # https://github.com/numpy/numpy/issues/22745
    # gen = np.random.default_rng(seed=gen.integers(0, int_info.max, num_clusters)[-1])
    # gen = np.random.default_rng(seed=gen.integers(0, int_info.max, trial+1)[-1])
    all_images = list(is_in_training.keys())
    clusters = gen.integers(low=0, high=num_clusters, size=len(all_images))
    cluster_map = {image: clusters[index] for index, image in enumerate(all_images)}
    reverse_cluster_map = defaultdict(list)
    failure_map = {
        row['image_file']: row['failed'] for index, row in test_data.iterrows()
    }
    failure_map.update({
        row['image_file']: row['failed'] for index, row in train_data.iterrows()
    })
    for image, index in cluster_map.items():
        reverse_cluster_map[index].append(image)
    num_actual_clusters = len(set(list(reverse_cluster_map.keys())))
    map_seen = {
        index: any([is_in_training[image] for image in images]) for index, images in reverse_cluster_map.items()
    }
    test_data[f'weighted_random_{trial}_seen'] = test_data.apply(lambda x: map_seen[cluster_map[x.image_file]],axis=1)
    seen_failures = len(test_data.loc[test_data['failed'] & test_data[f'weighted_random_{trial}_seen']])
    unseen_tests = len(test_data.loc[test_data[f'weighted_random_{trial}_seen'] == False])
    unseen_test_failures = len(
        test_data.loc[test_data['failed'] & (test_data[f'weighted_random_{trial}_seen'] == False)])
    total_seen = len(test_data.loc[test_data[f'weighted_random_{trial}_seen']])

    num_new_failures = 0
    num_new_failures_covered = 0
    new_failure_classes = set()
    new_failure_classes_covered = set()
    for index, row in test_data.iterrows():
        cluster_key = cluster_map[row.image_file]
        if row['failed']:
            training_data_from_cluster = train_data.loc[train_data['image_file'].isin(reverse_cluster_map[cluster_key])]
            training_data_from_cluster_failed = training_data_from_cluster[training_data_from_cluster['failed'] == True]
            if len(training_data_from_cluster_failed) == 0:
                num_new_failures += 1
                new_failure_classes.add(cluster_key)
                if len(training_data_from_cluster) > 0:
                    num_new_failures_covered += 1
                    new_failure_classes_covered.add(cluster_key)
    num_new_failure_classes = len(new_failure_classes)
    num_new_failure_classes_covered = len(new_failure_classes_covered)
    num_non_singleton_classes = 0
    num_consistent_non_singleton_classes = 0
    num_classes_with_failures = 0
    num_classes_with_all_failures = 0
    for cluster_key, images in reverse_cluster_map.items():
        any_fail = any([failure_map[image] for image in images])
        all_fail = all([failure_map[image] for image in images])
        if any_fail:
            num_classes_with_failures += 1
        if all_fail:
            num_classes_with_all_failures += 1
        if len(images) > 1:
            num_non_singleton_classes += 1
            # if any is True and all is True, then they all consistently failed
            # if any is False and all is False, then they all consistently succeeded
            # if any is True and all is False, then they were inconsistent
            # it is not possible for any to be False and all to be True
            if any_fail == all_fail:
                num_consistent_non_singleton_classes += 1


    perc_unseen_tests_failures = unseen_test_failures / unseen_tests if unseen_tests > 0 else np.nan
    failure_unseen_perc = unseen_test_failures / fail_count if fail_count > 0 else np.nan
    mix = perc_unseen_tests_failures * failure_unseen_perc
    return [trial, total_seen, total_seen / len_test,
           seen_failures, seen_failures / fail_count,
           unseen_test_failures, unseen_tests,
           perc_unseen_tests_failures, mix, num_actual_clusters,
           num_new_failures, num_new_failures_covered,
           num_new_failure_classes, num_new_failure_classes_covered,
           num_non_singleton_classes, num_consistent_non_singleton_classes,
           sum([1 if failed else 0 for image, failed in failure_map.items()]),
            num_classes_with_failures,
            num_classes_with_all_failures
           ]


def calc_baselines(train_data, test_data, output_path, log_file, random_trials=10, random_seed=1533):
    fail_count = len(test_data.loc[test_data['failed']])
    precision_df = pd.DataFrame({
        'precision': pd.Series(dtype='float'),
        'seen': pd.Series(dtype='int'),
        'seen_perc': pd.Series(dtype='float'),
        'fail_seen': pd.Series(dtype='int'),
        'fail_perc': pd.Series(dtype='float'),
        'unseen_test_fails': pd.Series(dtype='int'),
        'unseen_tests': pd.Series(dtype='int'),
        'perc_unseen_tests_fail': pd.Series(dtype='float'),
        'mix': pd.Series(dtype='float'),
        'num_clusters': pd.Series(dtype='int'),
        'unseen_mse': pd.Series(dtype='float'),
        'seen_mse': pd.Series(dtype='float'),
        'max_squared_error': pd.Series(dtype='float'),
        'total_mse': pd.Series(dtype='float'),
        'num_frames': pd.Series(dtype='int'),
        'num_new_failures': pd.Series(dtype='int'),
        'num_new_failures_covered': pd.Series(dtype='int'),
        'num_new_failure_classes': pd.Series(dtype='int'),
        'num_new_failure_classes_covered': pd.Series(dtype='int'),
        'num_non_singleton_classes': pd.Series(dtype='int'),
        'num_consistent_non_singleton_classes': pd.Series(dtype='int'),
        'total_failures': pd.Series(dtype='int'),
        'num_classes_with_failures': pd.Series(dtype='int'),
        'num_classes_with_all_failures': pd.Series(dtype='int')
    })
    len_test = len(test_data)
    label_precision = [1e-5 * i for i in range(1000)]
    label_precision.extend([1000 * 1e-5 + i * 1e-4 for i in range(1000)])
    label_precision.extend([1000 * 1e-4 + i * 1e-3 for i in range(1000)])
    print('Computing label precision')
    with Pool() as pool:
        results = {}
        print('Dispatching label jobs')
        for precision in tqdm(label_precision):
            # output_vector = calculate_label_baseline(fail_count, precision, test_data, train_data, len_test)
            results[precision] = pool.apply_async(calculate_label_baseline, (fail_count, precision, test_data, train_data, len_test))
        print('Waiting for results')
        for precision, result in tqdm(results.items()):
            precision_df.loc[len(precision_df.index)] = result.get()
    precision_df.to_csv(f'{output_path}/label_groups.csv')
    # calculate true random baseline
    is_in_training = {}
    for train_image in train_data['image_file'].values:
        is_in_training[train_image] = True
    for test_image in test_data['image_file'].values:
        is_in_training[test_image] = False
    gen = np.random.default_rng(seed=random_seed)
    weighted_random_df = pd.DataFrame({
        'trial': pd.Series(dtype='int'),
        'seen': pd.Series(dtype='int'),
        'seen_perc': pd.Series(dtype='float'),
        'fail_seen': pd.Series(dtype='int'),
        'fail_perc': pd.Series(dtype='float'),
        'unseen_test_fails': pd.Series(dtype='int'),
        'unseen_tests': pd.Series(dtype='int'),
        'perc_unseen_tests_fail': pd.Series(dtype='float'),
        'mix': pd.Series(dtype='float'),
        'num_clusters': pd.Series(dtype='int'),
        'num_new_failures': pd.Series(dtype='int'),
        'num_new_failures_covered': pd.Series(dtype='int'),
        'num_new_failure_classes': pd.Series(dtype='int'),
        'num_new_failure_classes_covered': pd.Series(dtype='int'),
        'num_non_singleton_classes': pd.Series(dtype='int'),
        'num_consistent_non_singleton_classes': pd.Series(dtype='int'),
        'total_failures': pd.Series(dtype='int'),
        'num_classes_with_failures': pd.Series(dtype='int'),
        'num_classes_with_all_failures': pd.Series(dtype='int')
    })
    all_images = list(is_in_training.keys())
    nums_to_check = [1]
    prev_cur_num = [1]
    cur_num = 2
    num_frames = len(all_images)

    def get_exp_val(cluster_count):
        return cluster_count - cluster_count * np.power((cluster_count - 1) / cluster_count, num_frames)
    prev = get_exp_val(cur_num)
    while get_exp_val(cur_num) < .99 * num_frames:
        prev = get_exp_val(cur_num)
        if prev >= prev_cur_num[-1] + 100:
            nums_to_check.append(cur_num)
            prev_cur_num.append(prev)
            # print(cur_num, prev)
        cur_num += 1
    print(f'Calculating {len(nums_to_check)} choices for true random')
    results = []
    index = 0
    with Pool(30) as p:
        print('Dispatching jobs')
        for num_clusters in tqdm(nums_to_check):
            if num_clusters < 1:
                continue
            for trial in range(1):
                index += 1
                r = p.apply_async(compute_true_random, (num_clusters, train_data, test_data, is_in_training, trial,
                                                        fail_count, len_test, random_seed, index))
                results.append(r)
                # r = compute_true_random(num_clusters, test_data, is_in_training, trial,
                #                                         fail_count, len_test, random_seed)
                # weighted_random_df.loc[len(weighted_random_df.index)] = r
        print('Waiting for jobs')
        for r in tqdm(results):
            weighted_random_df.loc[len(weighted_random_df.index)] = r.get()

    weighted_random_df.to_csv(f'{output_path}/true_random.csv')


def calculate_label_baseline(fail_count, precision, test_data, train_data, len_test):
    if precision == 0:  # 0 means exact match
        train_data[f'label_group_p{precision}'] = train_data.apply(lambda x: x.label_deg, axis=1)
        test_data[f'label_group_p{precision}'] = test_data.apply(lambda x: x.label_deg, axis=1)
    else:
        train_data[f'label_group_p{precision}'] = train_data.apply(lambda x: int(x.label_deg / precision), axis=1)
        test_data[f'label_group_p{precision}'] = test_data.apply(lambda x: int(x.label_deg / precision), axis=1)
    test_data[f'label_group_p{precision}_seen'] = test_data.apply(
        lambda x: x[f'label_group_p{precision}'] in train_data[f'label_group_p{precision}'].values, axis=1)
    clusters = pd.concat([train_data[f'label_group_p{precision}'], test_data[f'label_group_p{precision}']]).unique()
    num_clusters = len(clusters)
    unseen_mse = np.mean(test_data.loc[test_data[f'label_group_p{precision}_seen'] == False]['squared_error'])
    seen_mse = np.mean(test_data.loc[test_data[f'label_group_p{precision}_seen'] == True]['squared_error'])
    max_squared_error = np.max(test_data['squared_error'])
    total_mse = np.mean(test_data['squared_error'])
    seen_failures = len(test_data.loc[test_data['failed'] & test_data[f'label_group_p{precision}_seen']])
    unseen_tests = len(test_data.loc[test_data[f'label_group_p{precision}_seen'] == False])
    unseen_test_failures = len(
        test_data.loc[test_data['failed'] & (test_data[f'label_group_p{precision}_seen'] == False)])
    total_seen = len(test_data.loc[test_data[f'label_group_p{precision}_seen']])
    num_new_failures = 0
    num_new_failures_covered = 0
    new_failure_classes = set()
    new_failure_classes_covered = set()
    for index, row in test_data.iterrows():
        cluster_key = row[f'label_group_p{precision}']
        if row['failed']:
            training_data_from_cluster = train_data.loc[train_data[f'label_group_p{precision}'] == cluster_key]
            training_data_from_cluster_failed = training_data_from_cluster[
                training_data_from_cluster['failed'] == True]
            if len(training_data_from_cluster_failed) == 0:
                num_new_failures += 1
                new_failure_classes.add(cluster_key)
                if len(training_data_from_cluster) > 0:
                    num_new_failures_covered += 1
                    new_failure_classes_covered.add(cluster_key)
    num_new_failure_classes = len(new_failure_classes)
    num_new_failure_classes_covered = len(new_failure_classes_covered)
    num_non_singleton_classes = 0
    num_consistent_non_singleton_classes = 0
    num_classes_with_failures = 0
    num_classes_with_all_failures = 0
    for cluster_key in clusters:
        train_data_in_cluster = train_data.loc[train_data[f'label_group_p{precision}'] == cluster_key]
        test_data_in_cluster = test_data.loc[test_data[f'label_group_p{precision}'] == cluster_key]
        any_fail = any([train_data_in_cluster['failed'].any(), test_data_in_cluster['failed'].any()])
        all_fail = all([train_data_in_cluster['failed'].all(), test_data_in_cluster['failed'].all()])
        if any_fail:
            num_classes_with_failures += 1
        if all_fail:
            num_classes_with_all_failures += 1
        if len(train_data_in_cluster) + len(test_data_in_cluster) > 1:
            num_non_singleton_classes += 1
            # if any is True and all is True, then they all consistently failed
            # if any is False and all is False, then they all consistently succeeded
            # if any is True and all is False, then they were inconsistent
            # it is not possible for any to be False and all to be True
            if any_fail == all_fail:
                num_consistent_non_singleton_classes += 1
    perc_unseen_tests_failures = unseen_test_failures / unseen_tests if unseen_tests > 0 else np.nan
    failure_unseen_perc = unseen_test_failures / fail_count if fail_count > 0 else np.nan
    mix = perc_unseen_tests_failures * failure_unseen_perc
    return [precision, total_seen, total_seen/len_test, seen_failures,
                                                     seen_failures/fail_count, unseen_test_failures, unseen_tests,
                                                     perc_unseen_tests_failures, mix, num_clusters, unseen_mse,
                                                     seen_mse, max_squared_error, total_mse,
                                                     len(test_data) + len(train_data),
                                                     num_new_failures, num_new_failures_covered,
                                                     num_new_failure_classes, num_new_failure_classes_covered,
                                                     num_non_singleton_classes, num_consistent_non_singleton_classes,
                                                     len(test_data.loc[test_data['failed']]) +
                                                     len(train_data.loc[train_data['failed']]),
                                                     num_classes_with_failures,
                                                     num_classes_with_all_failures]


def parse_data(arg_string):
    args = custom_argparse(arg_string)
    output_path = args.output_path/''
    os.makedirs(str(output_path.absolute()), exist_ok=True)
    dataset = Dataset.load_from_file(args.dataset_file, '')
    image_to_cluster = dataset.image_to_cluster_map()
    cluster_to_image = defaultdict(list)
    for image, cluster in image_to_cluster.items():
        cluster_to_image[cluster].append(image)
    failure_thresh = args.failure_thresh_deg / (2 * args.deg_factor)
    log_file = f'{output_path}/log.txt'
    if os.path.exists(log_file):
        os.remove(log_file)
    if len(args.test_results) > 1:
        tests_to_check = args.test_results
    elif len(args.test_results) == 1:
        if args.test_results[0].is_dir():
            tests_to_check = [x for x in args.test_results[0].iterdir() if x.suffix == '.csv' and 'results' in x.stem]
        else:
            tests_to_check = args.test_results
    else:
        raise ValueError('')
    for test_file in tqdm(tests_to_check):
        training_town = test_file.stem[:test_file.stem.rfind('_')]
        print(training_town)
        full_data = pd.read_csv(test_file)
        full_data['label_deg'] = full_data.apply(lambda x: args.deg_factor * x.label, axis=1)
        full_data['pred_deg'] = full_data.apply(lambda x: args.deg_factor * x.pred, axis=1)
        full_data['abs_error'] = full_data.apply(lambda x: abs(x.label - x.pred), axis=1)
        full_data['abs_error_deg'] = full_data.apply(lambda x: args.deg_factor * abs(x.label - x.pred), axis=1)
        full_data['squared_error'] = full_data.apply(lambda x: np.power(x.label - x.pred, 2), axis=1)
        full_data['failed'] = full_data.apply(lambda x: abs(x.label - x.pred) > failure_thresh, axis=1)
        full_data['image_file'] = full_data.apply(lambda x: Path(f'{x.town}/{x.filename}'), axis=1)
        # # this is a temporary fix because I had to manually remove some of the Town10 data. This should be removed later
        # full_data = full_data.loc[full_data.apply(lambda x: x.image_file in image_to_cluster, axis=1)]
        assert(len(dataset) == len(full_data))
        full_data['cluster_key'] = full_data.apply(lambda x: image_to_cluster[x.image_file], axis=1)
        # plt.scatter(full_data['label_deg'], full_data['abs_error_deg'])
        # plt.xlabel('label deg')
        # plt.ylabel('abs error deg')
        # plt.show()
        if args.filter_deg > 0:
            full_data = full_data.loc[abs(full_data['label_deg']) <= args.filter_deg]
        test_data = full_data.loc[full_data['split'] == 'test'].copy()
        train_data = full_data.loc[full_data['split'] == 'train']
        training_map = {
            row['image_file']: row['split'] == 'train' for index, row in full_data.iterrows()
        }
        print(f'{len(train_data.loc[train_data["failed"]])} failed training points')
        print(f'{len(test_data.loc[test_data["failed"]])} failed testing points')
        if 'carla_abstract.json' in str(args.dataset_file):
            calc_baselines(train_data.copy(), test_data.copy(), output_path, log_file)
        print_file(f'Data is split into {len(train_data)} ({100*len(train_data) / len(full_data):.1f}%) train and {len(test_data)} test ({100*len(test_data) / len(full_data):.1f}%) of {len(full_data)} total', log_file)
        filter_and_save_clusters(dataset, full_data, output_path, test_data, train_data)
        cluster_to_train_count = defaultdict(int)
        clusters_in_training = set()
        clusters_in_testing = set()
        for index, row in train_data.iterrows():
            cluster_key = image_to_cluster[row.image_file]
            cluster_to_train_count[cluster_key] += 1
            clusters_in_training.add(cluster_key)
        num_new_failures = 0
        num_new_failures_covered = 0
        new_failure_classes = set()
        new_failure_classes_covered = set()
        for index, row in test_data.iterrows():
            cluster_key = image_to_cluster[row.image_file]
            clusters_in_testing.add(cluster_key)
            if row['failed']:
                training_data_from_cluster = train_data.loc[train_data['cluster_key'] == cluster_key]
                training_data_from_cluster_failed = training_data_from_cluster[training_data_from_cluster['failed'] == True]
                if len(training_data_from_cluster_failed) == 0:
                    num_new_failures += 1
                    new_failure_classes.add(cluster_key)
                    if len(training_data_from_cluster) > 0:
                        num_new_failures_covered += 1
                        new_failure_classes_covered.add(cluster_key)
        num_new_failure_classes = len(new_failure_classes)
        num_new_failure_classes_covered = len(new_failure_classes_covered)
        only_in_test = clusters_in_testing.difference(clusters_in_training)
        only_in_train = clusters_in_training.difference(clusters_in_testing)
        clusters_in_both = clusters_in_training.intersection(clusters_in_testing)

        print_file(f'Only in test: {len(only_in_test)} clusters containing {sum([len(dataset._clusters[c]) for c in only_in_test])} images', log_file)
        print_file(f'Only in train: {len(only_in_train)} clusters containing {sum([len(dataset._clusters[c]) for c in only_in_train])} images', log_file)
        print_file(f'In both: {len(clusters_in_both)} clusters containing {sum([len(dataset._clusters[c]) for c in clusters_in_both])} images', log_file)
        test_data['seen_in_training'] = test_data.apply(lambda x: x.cluster_key in clusters_in_training, axis=1)
        failed_tests = test_data.loc[test_data['failed']]
        failed_tests.to_csv(f'{output_path}/failed.csv')
        if args.save_images:
            seen_folder = f'{output_path}/seen/'
            unseen_folder = f'{output_path}/unseen/'
            for folder in [seen_folder, unseen_folder]:
                os.makedirs(folder, exist_ok=True)
            print('Saving images, this may take a moment')
            for index, row in tqdm(failed_tests.iterrows(), total=len(failed_tests)):
                image_file = f'{args.input_path}/{row.image_file}'
                image = cv2.imread(image_file)
                image = cv2.putText(image, f'label: {row.label * 140:.2f}', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 0), 2, cv2.LINE_AA)
                image = cv2.putText(image, f'pred: {row.pred * 140:.2f}', (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 0), 2, cv2.LINE_AA)
                image = cv2.putText(image, f'err: {row.abs_error * 140:.2f}', (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 0),
                                    2,
                                    cv2.LINE_AA)
                image = cv2.putText(image, 'Seen' if row.seen_in_training else 'Unseen', (0, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),
                                    2,
                                    cv2.LINE_AA)
                save_folder = seen_folder if row.seen_in_training else unseen_folder
                save_image = str(row.image_file).replace("/", "_")
                cv2.imwrite(f'{save_folder}{save_image}', image)
        if args.save_all:
            train_dir = f'{output_path}/train_images/'
            test_dir = f'{output_path}/test_images/'
            for folder in [train_dir, test_dir]:
                os.makedirs(folder, exist_ok=True)
            for index, row in tqdm(full_data.iterrows(), total=len(full_data)):
                image_file = f'{args.input_path}/{row.image_file}'
                image = cv2.imread(image_file)
                image = cv2.putText(image, f'label: {row.label * 140:.2f}', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 0), 2, cv2.LINE_AA)
                image = cv2.putText(image, f'pred: {row.pred * 140:.2f}', (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 0), 2, cv2.LINE_AA)
                image = cv2.putText(image, f'err: {row.abs_error * 140:.2f}', (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 0),
                                    2,
                                    cv2.LINE_AA)
                image = cv2.putText(image, 'Fail' if row.failed else 'Success', (0, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),
                                    2,
                                    cv2.LINE_AA)
                image = cv2.putText(image, row.split, (0, 250),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),
                                    2,
                                    cv2.LINE_AA)
                save_folder = train_dir if row.split == 'train' else test_dir
                save_image = str(row.image_file).replace("/", "_")
                cv2.imwrite(f'{save_folder}{save_image}', image)
        failed_train = train_data.loc[train_data['failed']]
        failed_full = full_data.loc[full_data['failed']]
        failure_map = {
            row['image_file']: row['failed'] for index, row in full_data.iterrows()
        }
        number_in_training = {}  # map from cluster key to # of times that cluster is seen in training
        number_tests_failed = {}
        number_in_tests = {}
        train_counts = train_data['cluster_key'].value_counts()
        test_counts = test_data['cluster_key'].value_counts()
        failed_test_counts = failed_tests['cluster_key'].value_counts()
        for cluster_key in dataset._clusters:
            number_in_training[cluster_key] = train_counts[cluster_key] if cluster_key in train_counts else 0
            number_tests_failed[cluster_key] = failed_test_counts[cluster_key] if cluster_key in failed_test_counts else 0
            number_in_tests[cluster_key] = test_counts[cluster_key] if cluster_key in test_counts else 0
            # number_in_training[cluster_key] = len(train_data.loc[train_data['cluster_key'] == cluster_key])
            # number_tests_failed[cluster_key] = len(failed_tests.loc[failed_tests['cluster_key'] == cluster_key])
        sorted_keys = sorted([key for key, value in number_tests_failed.items()],
                             key=lambda x: (number_in_training[x], number_tests_failed[x]), reverse=True)
        sorted_keys_fail_only = sorted([key for key, value in number_tests_failed.items() if value > 0],
                             key=lambda x: (number_in_training[x], number_tests_failed[x]), reverse=True)
        fail_df = pd.DataFrame(data={
            'num_train': [number_in_training[key] for key in sorted_keys],
            'num_test_fail': [number_tests_failed[key] for key in sorted_keys],
            'num_test': [number_in_tests[key] for key in sorted_keys],
            'test_fail_perc': [(number_tests_failed[key] / number_in_tests[key]) if number_in_tests[key] > 0 else 0 for
                               key in sorted_keys],
        })
        fail_df_fail_only = pd.DataFrame(data={
            'num_train': [number_in_training[key] for key in sorted_keys_fail_only],
            'num_test_fail': [number_tests_failed[key] for key in sorted_keys_fail_only],
            'num_test': [number_in_tests[key] for key in sorted_keys_fail_only],
            'test_fail_perc': [(number_tests_failed[key] / number_in_tests[key]) if number_in_tests[key] > 0 else 0 for
                               key in sorted_keys_fail_only],
        })
        print_file(f'Failures in test: {len(failed_tests)}/{len(test_data)} ({100.0 * len(failed_tests) / len(test_data):.2f}%)', log_file)
        print_file(f'Failures in train: {len(failed_train)}/{len(train_data)} ({100.0 * len(failed_train) / len(train_data):.2f}%)', log_file)
        print_file(f'Failures in total: {len(failed_full)}/{len(full_data)} ({100.0 * len(failed_full) / len(full_data):.2f}%)', log_file)
        test_data_seen = test_data.loc[test_data['cluster_key'].isin(clusters_in_both)]
        test_data_unseen = test_data.loc[test_data['cluster_key'].isin(only_in_test)]
        test_failures_on_seen_data = test_data_seen.loc[test_data_seen['failed']]
        test_failures_on_unseen_data = test_data_unseen.loc[test_data_unseen['failed']]
        clusters = pd.concat([train_data['cluster_key'], test_data['cluster_key']]).unique()
        num_clusters = len(clusters)
        unseen_mse = np.mean(test_data_unseen['squared_error'])
        seen_mse = np.mean(test_data_seen['squared_error'])
        max_squared_error = np.max(test_data['squared_error'])
        total_mse = np.mean(test_data['squared_error'])
        print_file(f'Failures on unseen data: {len(test_failures_on_unseen_data)} failures of {len(test_data_unseen)} unseen test data ({100*len(test_failures_on_unseen_data) / len(test_data_unseen) if len(test_data_unseen) > 0 else np.nan:.2f}%)', log_file)
        print_file(f'Failures on seen data: {len(test_failures_on_seen_data)} failures of {len(test_data_seen)} seen test data ({100 * len(test_failures_on_seen_data) / len(test_data_seen) if len(test_data_seen) > 0 else 0:.2f}%)', log_file)
        print_file(f'% of failures that are unseen: {len(test_failures_on_unseen_data)} failures of {len(failed_tests)} unseen test data ({100*len(test_failures_on_unseen_data) / len(failed_tests) if len(failed_tests) > 0 else np.nan:.2f}%)', log_file)
        graph_df = pd.DataFrame({
            'file': pd.Series(dtype='string'),
            'seen': pd.Series(dtype='int'),
            'seen_perc': pd.Series(dtype='float'),
            'fail_seen': pd.Series(dtype='int'),
            'fail_perc': pd.Series(dtype='float'),
            'unseen_test_fails': pd.Series(dtype='int'),
            'unseen_tests': pd.Series(dtype='int'),
            'perc_unseen_tests_fail': pd.Series(dtype='float'),
            'mix': pd.Series(dtype='float'),
            'num_clusters': pd.Series(dtype='int'),
            'unseen_mse': pd.Series(dtype='float'),
            'seen_mse': pd.Series(dtype='float'),
            'max_squared_error': pd.Series(dtype='float'),
            'total_mse': pd.Series(dtype='float'),
            'num_new_failures': pd.Series(dtype='int'),
            'num_new_failures_covered': pd.Series(dtype='int'),
            'num_new_failure_classes': pd.Series(dtype='int'),
            'num_new_failure_classes_covered': pd.Series(dtype='int'),
            'num_non_singleton_classes': pd.Series(dtype='int'),
            'num_consistent_non_singleton_classes': pd.Series(dtype='int'),
            'total_failures': pd.Series(dtype='int'),
            'num_classes_with_failures': pd.Series(dtype='int'),
            'num_classes_with_all_failures': pd.Series(dtype='int')
        })
        perc_unseen_tests_failures = len(test_failures_on_unseen_data) / len(test_data_unseen) if len(test_data_unseen) > 0 else np.nan
        failure_unseen_perc = len(test_failures_on_unseen_data)/len(failed_tests) if len(failed_tests) > 0 else np.nan
        mix = perc_unseen_tests_failures * failure_unseen_perc

        num_non_singleton_classes = 0
        num_consistent_non_singleton_classes = 0
        num_classes_with_failures = 0
        num_classes_with_all_failures = 0
        print('Checking homogeneity')
        for cluster_key in tqdm(clusters):
            images = cluster_to_image[cluster_key]
            any_fail = any([failure_map[image] for image in images])
            all_fail = all([failure_map[image] for image in images])
            if any_fail:
                num_classes_with_failures += 1
            if all_fail:
                num_classes_with_all_failures += 1
            if len(images) > 1:
                num_non_singleton_classes += 1
                # if any is True and all is True, then they all consistently failed
                # if any is False and all is False, then they all consistently succeeded
                # if any is True and all is False, then they were inconsistent
                # it is not possible for any to be False and all to be True
                if any_fail == all_fail:
                    num_consistent_non_singleton_classes += 1

        graph_df.loc[len(graph_df.index)] = [training_town, len(test_data_seen), len(test_data_seen)/len(test_data), len(test_failures_on_seen_data),
                                             len(test_failures_on_seen_data)/len(failed_tests), len(test_failures_on_unseen_data), len(test_data_unseen), perc_unseen_tests_failures, mix,
                                             num_clusters, unseen_mse, seen_mse, max_squared_error, total_mse,
                                             num_new_failures, num_new_failures_covered,
                                             num_new_failure_classes, num_new_failure_classes_covered,
                                             num_non_singleton_classes, num_consistent_non_singleton_classes,
                                             len(failed_full),
                                             num_classes_with_failures,
                                             num_classes_with_all_failures
                                             ]
        graph_df.to_csv(f'{output_path}/seen_stats.csv')
        clusters_with_training = set()
        clusters_with_tests = set()
        clusters_with_failures = set()
        clusters_with_pass = set()
        clusters_with_test_pass = set()
        clusters_with_test_fail = set()
        clusters_with_train_pass = set()
        clusters_with_train_fail = set()
        for cluster_key in clusters:
            images = cluster_to_image[cluster_key]
            any_fail = any([failure_map[image] for image in images])
            all_fail = all([failure_map[image] for image in images])
            any_train = any([training_map[image] for image in images])
            all_train = all([training_map[image] for image in images])
            if any_train:
                clusters_with_training.add(cluster_key)
            if not all_train:
                # this means some were test
                clusters_with_tests.add(cluster_key)
            if any_fail:
                clusters_with_failures.add(cluster_key)
            if not all_fail:
                # this means some passed
                clusters_with_pass.add(cluster_key)
            if any([failure_map[image] and training_map[image] for image in images]):
                clusters_with_train_fail.add(cluster_key)
            if any([(not failure_map[image]) and training_map[image] for image in images]):
                clusters_with_train_pass.add(cluster_key)
            if any([failure_map[image] and (not training_map[image]) for image in images]):
                clusters_with_test_fail.add(cluster_key)
            if any([(not failure_map[image]) and (not training_map[image]) for image in images]):
                clusters_with_test_pass.add(cluster_key)
        labels = venn.get_labels(
            [clusters_with_training, clusters_with_tests, clusters_with_pass, clusters_with_failures],
            fill=['number', 'logic'])
        fig, ax = venn.venn4(labels, names=['Train', 'Test', 'Pass', 'Fail'])
        fig.savefig(f'{output_path}/train_test_pass_fail_class_venn.png')
        labels = venn.get_labels(
            [clusters_with_train_pass, clusters_with_train_fail, clusters_with_test_pass, clusters_with_test_fail],
            fill=['number', 'logic'])
        fig, ax = venn.venn4(labels, names=['Train Pass', 'Train Fail', 'Test Pass', 'Test Fail'])
        fig.savefig(f'{output_path}/class_venn.png')
        intersection_1111 = clusters_with_train_fail.intersection(clusters_with_train_pass).intersection(clusters_with_test_fail).intersection(clusters_with_test_pass)
        print(intersection_1111)

        if len(failed_tests) > 0 and len(test_data_unseen) > 0:
            perc_unseen_tests_fail = len(test_failures_on_unseen_data) / len(test_data_unseen)
            perc_failures_on_unseen = len(test_failures_on_unseen_data) / len(failed_tests)
            combined_metric = perc_unseen_tests_fail * perc_failures_on_unseen
        else:
            combined_metric = np.nan
        print_file(f'Combined metric: {100*combined_metric:.2f}%', log_file)
        plot_failed_data(fail_df, training_town, args.failure_thresh_deg, output_path, filter_deg=args.filter_deg, log_file=log_file)
        return


def filter_and_save_clusters(dataset, full_data, output_path, test_data, train_data):
    # Save cluster splits:
    # train
    train_dataset = copy.deepcopy(dataset)
    train_dataset.filter_by_image_list(train_data['image_file'].tolist())
    assert(len(train_dataset) == len(train_data))
    train_dataset.save_to_file(f'{output_path}/train.json', '', '')
    # test
    test_dataset = copy.deepcopy(dataset)
    test_dataset.filter_by_image_list(test_data['image_file'].tolist())
    assert(len(test_dataset) == len(test_data))
    test_dataset.save_to_file(f'{output_path}/test.json', '', '')
    # train fail
    train_fail_dataset = copy.deepcopy(dataset)
    train_fail_dataset.filter_by_image_list(train_data.loc[train_data['failed']]['image_file'].tolist())
    assert(len(train_fail_dataset) == len(train_data.loc[train_data['failed']]))
    train_fail_dataset.save_to_file(f'{output_path}/train_fail.json', '', '')
    # test fail
    test_fail_dataset = copy.deepcopy(dataset)
    test_fail_dataset.filter_by_image_list(test_data.loc[test_data['failed']]['image_file'].tolist())
    assert(len(test_fail_dataset) == len(test_data.loc[test_data['failed']]))
    test_fail_dataset.save_to_file(f'{output_path}/test_fail.json', '', '')
    # train succ
    train_succ_dataset = copy.deepcopy(dataset)
    train_succ_dataset.filter_by_image_list(train_data.loc[train_data['failed'] == False]['image_file'].tolist())
    assert(len(train_succ_dataset) == len(train_data.loc[train_data['failed'] == False]))
    train_succ_dataset.save_to_file(f'{output_path}/train_succ.json', '', '')
    # test succ
    test_succ_dataset = copy.deepcopy(dataset)
    test_succ_dataset.filter_by_image_list(test_data.loc[test_data['failed'] == False]['image_file'].tolist())
    assert(len(test_succ_dataset) == len(test_data.loc[test_data['failed'] == False]))
    test_succ_dataset.save_to_file(f'{output_path}/test_succ.json', '', '')
    # succ
    succ_dataset = copy.deepcopy(dataset)
    succ_dataset.filter_by_image_list(full_data.loc[full_data['failed'] == False]['image_file'].tolist())
    assert(len(succ_dataset) == len(full_data.loc[full_data['failed'] == False]))
    succ_dataset.save_to_file(f'{output_path}/succ.json', '', '')
    # fail
    fail_dataset = copy.deepcopy(dataset)
    fail_dataset.filter_by_image_list(full_data.loc[full_data['failed']]['image_file'].tolist())
    assert(len(fail_dataset) == len(full_data.loc[full_data['failed']]))
    fail_dataset.save_to_file(f'{output_path}/fail.json', '', '')


if __name__ == '__main__':
    parse_data(sys.argv[1:])
