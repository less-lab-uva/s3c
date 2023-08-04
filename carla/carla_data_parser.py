import argparse
import sys
from collections import defaultdict

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
        "-test",
        "--test_results",
        type=Path,
        nargs='+',
        required=True,
        help="Results of model on tests."
    )
    parser.add_argument(
        "-f",
        "--failure_thresh",
        type=float,
        required=False,
        default=5.0/140.0,
        help="Failure threshold. Data ranges -1 to 1, default is 5/140 to approximate 5 deg"
    )
    return parser.parse_args(arg_string)

def get_frame_split(df, image_file):
    row = df.loc[df["image_file"] == image_file]
    if len(row) != 1:
        raise ValueError()
    return row.split.tolist()[0]

def parse_data(arg_string):
    args = custom_argparse(arg_string)
    dataset = Dataset.load_from_file(args.dataset_file, '')
    image_to_cluster = dataset.image_to_cluster_map()
    print(args.test_results)
    failed_x_by_town = {}
    failed_y_by_town = {}
    abs_error_by_town = {}
    mse_error_by_town = {}
    # succeeded_x_by_town = {}
    succeeded_y_by_town = {}
    for test_file in args.test_results:
        training_town = test_file.stem[:test_file.stem.rfind('_')] + '/'
        print(training_town)
        full_data = pd.read_csv(test_file)
        full_data['abs_error'] = full_data.apply(lambda x: abs(x.label - x.pred), axis=1)
        full_data['squared_error'] = full_data.apply(lambda x: np.power(x.label - x.pred, 2), axis=1)
        full_data['failed'] = full_data.apply(lambda x: abs(x.label - x.pred) > args.failure_thresh, axis=1)
        full_data['image_file'] = full_data.apply(lambda x: Path(f'{x.town}/{x.filename}'), axis=1)
        test_data = full_data.loc[full_data['split'] == 'test']
        train_data = full_data.loc[full_data['split'] == 'train']
        print(f'Data is split into {len(train_data)} ({100*len(train_data) / len(full_data):.1f}%) train and {len(test_data)} test ({100*len(test_data) / len(full_data):.1f}%) of {len(full_data)} total')
        cluster_to_train_count = defaultdict(int)
        clusters_in_training = set()
        clusters_in_testing = set()
        for index, row in train_data.iterrows():
            cluster_key = image_to_cluster[row.image_file]
            cluster_to_train_count[cluster_key] += 1
            clusters_in_training.add(cluster_key)
        for index, row in test_data.iterrows():
            cluster_key = image_to_cluster[row.image_file]
            clusters_in_testing.add(cluster_key)
        only_in_test = clusters_in_testing.difference(clusters_in_training)
        only_in_train = clusters_in_training.difference(clusters_in_testing)
        print(f'Only in test: {len(only_in_test)} clusters containing {sum([len(dataset._clusters[c]) for c in only_in_test])} images')
        print(f'Only in train: {len(only_in_train)} clusters containing {sum([len(dataset._clusters[c]) for c in only_in_train])} images')
        only_failed = test_data.loc[test_data['failed']]
        print(f'Total failures: {len(only_failed)}/{len(test_data)} ({100.0 * len(only_failed)/len(test_data):.2f}%)')
        # for image in dataset._image_files:
        #     if 'Town02' in image:
        #         print(image)
        failed_count_dict = defaultdict(int)
        succeeded_count_dict = defaultdict(int)
        abs_error_dict = defaultdict(list)
        mse_error_dict = defaultdict(list)
        for index, row in test_data.iterrows():
            # print(row.town, row.filename)
            # print(image_file in dataset._image_files)
            cluster_key = image_to_cluster[row.image_file]
            # cluster = dataset._clusters[cluster_key]
            # num_in_training = len([1 for image in cluster if image in train_data['image_file'].values])
            num_in_training = cluster_to_train_count[cluster_key]
            (failed_count_dict if row.failed else succeeded_count_dict)[num_in_training] += 1
            abs_error_dict[num_in_training].append(row.abs_error)
            mse_error_dict[num_in_training].append(row.squared_error)
        failed_x = []
        abs_errors = []
        mse_errors = []
        failed_y = []
        succeeded_y = []
        # print(training_town)
        for num_in_training in sorted(failed_count_dict.keys()):
            x = num_in_training
            failed_x.append(x)
            failed_y.append(failed_count_dict[num_in_training])
            # succeeded_x_by_town.append(x)
            succeeded_y.append(succeeded_count_dict[num_in_training])
            abs_errors.append(abs_error_dict[num_in_training])
            mse_errors.append(mse_error_dict[num_in_training])
            print(f'{num_in_training}\t{failed_count_dict[num_in_training]}\t{succeeded_count_dict[num_in_training]}\t{100.0*failed_count_dict[num_in_training] / (failed_count_dict[num_in_training] + succeeded_count_dict[num_in_training]):.2f}%')
#            print(len(abs_error_dict[num_in_training]))
        confusion = np.array([[0,0],[0,0]])
        confusion[0,0] = failed_count_dict[0] if 0 in failed_count_dict else 0
        confusion[1,0] = sum([failed_count_dict[num_in_training] for num_in_training in failed_count_dict.keys() if num_in_training != 0])
        confusion[0,1] = succeeded_count_dict[0] if 0 in succeeded_count_dict else 0
        confusion[1,1] = sum([succeeded_count_dict[num_in_training] for num_in_training in succeeded_count_dict.keys() if num_in_training != 0])
        conf_sum = np.sum(confusion)
        print(training_town)
        print('_\t# Un\t# Seen\tTotal')
        print(f'Fail\t{confusion[0, 0]} ({100 * confusion[0, 0] / conf_sum:0.1f}%)\t{confusion[1, 0]} ({100 * confusion[1, 0] / conf_sum:0.1f}%)\t{confusion[0, 0] + confusion[1, 0]} ({100 * (confusion[0, 0] + confusion[1, 0]) / conf_sum:0.1f}%)')
        print(f'Succ\t{confusion[0, 1]} ({100 * confusion[0, 1] / conf_sum:0.1f}%)\t{confusion[1, 1]} ({100 * confusion[1, 1] / conf_sum:0.1f}%)\t{confusion[0, 1] + confusion[1, 1]} ({100 * (confusion[0, 1] + confusion[1, 1]) / conf_sum:0.1f}%)')
        print(f'Total\t{confusion[0, 0] + confusion[0, 1]} ({100 * (confusion[0, 0] + confusion[0, 1]) / conf_sum:0.1f}%)\t{confusion[1, 0] + confusion[1, 1]} ({100*(confusion[1, 0] + confusion[1, 1]) / conf_sum:0.1f}%)\t{conf_sum}')
        print('Unseen vs Seen splits')
        # print(abs_error_dict[0])
        unseen_abs = [item for item in abs_error_dict[0]] if 0 in abs_error_dict else []
        #print('seen abs')
        #print([abs_error_dict[num_in_training] for num_in_training in failed_count_dict.keys() if num_in_training != 0])
        seen_abs = [item for item in abs_error_dict[num_in_training] for num_in_training in failed_count_dict.keys() if num_in_training != 0]
        unseen_mse = [item for item in mse_error_dict[0]] if 0 in mse_error_dict else []
        seen_mse = [item for item in mse_error_dict[num_in_training] for num_in_training in failed_count_dict.keys() if num_in_training != 0]
        print(f'Num: {len(unseen_abs)}\t{len(seen_abs)}')
        #print(unseen_abs)
        # print([type(a) for a in unseen_abs])
        # print('len 0', len(unseen_abs[0]))
        # print(sum(unseen_abs))
        print(f'Total abs error: {sum(unseen_abs)}\t{sum(seen_abs)}')
        print(f'Avg abs error: {sum(unseen_abs)/len(unseen_abs) if len(unseen_abs) > 0 else np.nan}\t{sum(seen_abs)/len(seen_abs) if len(seen_abs) > 0 else np.nan}')
        print(f'MSE: {sum(unseen_mse)/len(unseen_mse) if len(unseen_mse) > 0 else np.nan}\t{sum(seen_mse)/len(seen_mse) if len(seen_mse) > 0 else np.nan}')
        failed_x_by_town[training_town] = failed_x
        failed_y_by_town[training_town] = failed_y
        abs_error_by_town[training_town] = abs_errors
        mse_error_by_town[training_town] = mse_errors
        succeeded_y_by_town[training_town] = succeeded_y
    failed_count_dict = defaultdict(int)
    succeeded_count_dict = defaultdict(int)
    abs_error_dict = defaultdict(list)
    mse_error_dict = defaultdict(list)
    for town in failed_x_by_town:
        for x, failed_y, succeeded_y, abs_error, mse_error in zip(failed_x_by_town[town], failed_y_by_town[town], succeeded_y_by_town[town], abs_error_by_town[town], mse_error_by_town[town]):
            failed_count_dict[x] += failed_y
            succeeded_count_dict[x] += succeeded_y
            abs_error_dict[x].extend(abs_error)
            mse_error_dict[x].extend(mse_error)
    if len(args.test_results) > 1:
        failed_x = []
        failed_y = []
        succeeded_y = []
        print('All')
        for num_in_training in sorted(failed_count_dict.keys()):
            x = num_in_training
            failed_x.append(x)
            failed_y.append(failed_count_dict[num_in_training])
            succeeded_y.append(succeeded_count_dict[num_in_training])
            print(f'{num_in_training}\t{failed_count_dict[num_in_training]}\t{succeeded_count_dict[num_in_training]}\t{100.0*failed_count_dict[num_in_training] / (failed_count_dict[num_in_training] + succeeded_count_dict[num_in_training]):.2f}%')
        failed_x_by_town['All'] = failed_x
        failed_y_by_town['All'] = failed_y
        succeeded_y_by_town['All'] = succeeded_y
        confusion = np.array([[0,0],[0,0]])
        confusion[0,0] = failed_count_dict[0] if 0 in failed_count_dict else 0
        confusion[1,0] = sum([failed_count_dict[num_in_training] for num_in_training in failed_count_dict.keys() if num_in_training != 0])
        confusion[0,1] = succeeded_count_dict[0] if 0 in succeeded_count_dict else 0
        confusion[1,1] = sum([succeeded_count_dict[num_in_training] for num_in_training in succeeded_count_dict.keys() if num_in_training != 0])
        conf_sum = np.sum(confusion)
        print('All')
        print('_\t# Un\t# Seen\tTotal')
        print(f'Fail\t{confusion[0, 0]} ({100 * confusion[0, 0] / conf_sum:0.1f}%)\t{confusion[1, 0]} ({100 * confusion[1, 0] / conf_sum:0.1f}%)\t{confusion[0, 0] + confusion[1, 0]} ({100 * (confusion[0, 0] + confusion[1, 0]) / conf_sum:0.1f}%)')
        print(f'Succ\t{confusion[0, 1]} ({100 * confusion[0, 1] / conf_sum:0.1f}%)\t{confusion[1, 1]} ({100 * confusion[1, 1] / conf_sum:0.1f}%)\t{confusion[0, 1] + confusion[1, 1]} ({100 * (confusion[0, 1] + confusion[1, 1]) / conf_sum:0.1f}%)')
        print(f'Total\t{confusion[0, 0] + confusion[0, 1]} ({100 * (confusion[0, 0] + confusion[0, 1]) / conf_sum:0.1f}%)\t{confusion[1, 0] + confusion[1, 1]} ({100 * (confusion[1, 0] + confusion[1, 1]) / conf_sum:0.1f}%)\t{conf_sum}')
        unseen_abs = [item for item in abs_error_dict[0]] if 0 in abs_error_dict else []
        seen_abs = [item for item in abs_error_dict[num_in_training] for num_in_training in failed_count_dict.keys() if num_in_training != 0]
        unseen_mse = [item for item in mse_error_dict[0]] if 0 in mse_error_dict else []
        seen_mse = [item for item in mse_error_dict[num_in_training] for num_in_training in failed_count_dict.keys() if num_in_training != 0]
        print(f'Num: {len(unseen_abs)}\t{len(seen_abs)}')
        print(f'Total abs error: {sum(unseen_abs)}\t{sum(seen_abs)}')
        print(f'Avg abs error: {sum(unseen_abs)/len(unseen_abs) if len(unseen_abs) > 0 else np.nan}\t{sum(seen_abs)/len(seen_abs) if len(seen_abs) else np.nan}')
        print(f'MSE: {sum(unseen_mse)/len(unseen_mse) if len(unseen_mse) > 0 else np.nan}\t{sum(seen_mse)/len(seen_mse) if len(seen_mse) > 0 else np.nan}')
    loglog = False
    for town in failed_x_by_town:
        if town == 'All':
            continue
        percent = [100.0 * failed / (failed + succeeded) for failed, succeeded in zip(failed_y_by_town[town], succeeded_y_by_town[town])]
        if loglog:
            x = [x + 1 for x in failed_x_by_town[town]]
        else:
            x = failed_x_by_town[town]
        plt.scatter(x, failed_y_by_town[town], label=town)
        # plt.scatter(failed_x_by_town[town], percent, label=town)
        # plt.scatter(failed_x_by_town[town], succeeded_y_by_town[town], label=town)
    plt.legend()
    if loglog:
        plt.xscale('log')
        plt.yscale('log')
    plt.xlabel('Number of times observed in training')
    plt.ylabel('# failures')
    # plt.ylabel('% failures')
    # plt.ylabel('# Successes')
    plt.title('Failures vs Occurrences in training')
    plt.show()



if __name__ == '__main__':
    parse_data(sys.argv[1:])
