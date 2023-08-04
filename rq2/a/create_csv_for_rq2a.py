import time
import sys
sys.path.insert(0, '../../')

from utils.dataset import Dataset
from pathlib import Path

def main():
    start = time.perf_counter_ns()
    dataset_1 = Dataset.load_from_file("../../study_data/results/0.8_percent_e386/carla_rsv/test_fail.json", "PATH_TO_DATA", "PATH_TO_SGs")
    dataset_2 = Dataset.load_from_file("../../study_data/results/0.8_percent_e386/carla_rsv/train.json", "PATH_TO_DATA", "PATH_TO_SGs")

    # Check if output folder exists and creates it
    output_path = Path("./splits_csv/")
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    union_dataset, intersection_dataset, d1_diff_d2, d2_diff_d1 = Dataset.dataset_venn(
        dataset_1, dataset_2, threads=32, verbose=True)

    ### Save cluster###
    # 1 Diff 2
    d1_diff_d2_dataset_path = output_path / "test_fail_diff_train.json"
    d1_diff_d2_csv = output_path / "test_fail_diff_train.csv"

    d1_diff_d2.save_to_file(d1_diff_d2_dataset_path, "", "")
    d1_diff_d2.export_as_csv(
        d1_diff_d2_csv, "", verbose=True, threads=32)

    # 2 Diff 1
    d2_diff_d1_dataset_path = output_path / "train_diff_test_fail.json"
    d2_diff_d1_csv = output_path / "train_diff_test_fail.csv"

    d2_diff_d1.save_to_file(
        d2_diff_d1_dataset_path, "", "")
    d2_diff_d1.export_as_csv(d2_diff_d1_csv, "", verbose=True, threads=32)

    end = time.perf_counter_ns()
    print(f"Time for saving clusters and exporting CSVs: {(end-start)*1e-9}")


if __name__ == '__main__':
    main()
