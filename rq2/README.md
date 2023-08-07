# Scripts for generating and parsing data for RQ2
## Folder Structure
The RQ2 code is split across two different folders, `a` and `b`, for generating the data found in Sections 4.3.1 and 4.3.2 respectively.
The folder is organized as follows:
* `a/`
  * `splits_csv_orig/`
    * CSV files containing the graph features corresponding to the equivalence classes that either occurred in training and did not fail in testing or vice versa.
  * `create_csv_for_rq2a.py`
    * Given the full study data, this script generates the CSVs found in `splits_csv_orig/`.
  * `rq2a.py`
    * Run as part of `study_data/generate_figures.py`, this generates the decision tree shown in Figure 5. NOTE: `tree.png` is 30,000 by 30,000 pixels to allow for zooming in to read the predicates. This file may not load on some machines. Figure 5 shows `tree_small.png` (1,500 by 1,500 pixels) to visualize the structure of the tree.
* `b/`
  * `carla_csv_orig/`
    * CSV files containing the graph features needed to compute coverage of the preconditions studies in Section 4.3.2.
  * `create_csv_for_rq2b.py`
    * Given the full study data, this script generates the CSVs found in `carla_csv_orig/`.
  * `rq2b.py`
    * Run as part of `study_data/generate_figures.py`, this generates the precondition coverage information found in Table 2.

## Table
The data to fill the table is obtained by executing `rq2b.py`. The table is shown below for convenience. Note that in our paper, there was a typo in the table. The correct value for precondition 5 coverage of Town04 is 149 (34.98%), not 147 (34.51%).

| Preconditions | \|A\| | Town01     | Town02     | Town04        | Town10HD    | All          |
|---------------|-------|------------|------------|---------------|-------------|--------------|
| 1             | 242   | 19 (7.85%) | 5 (2.07%)  | 3 (1.24%)     | 6 (2.48%)   | 20 (8.26%)   |
| 2             | 31    | 0 (0.0%)   | 0 (0.0%)   | 24 (77.42%)   | 23 (74.19%) | 26 (83.87%)  |
| 3             | 24    | 0 (0.0%)   | 0 (0.0%)   | 0 (0.0%)      | 0 (0.0%)    | 0 (0.0%)     |
| 4             | 22    | 4 (18.18%) | 4 (18.18%) | 22 (100.0%)   | 10 (45.45%) | 22 (100.0%)  |
| 5             | 426   | 3 (0.7%)   | 3 (0.7%)   | **149 (34.98%)*** | 16 (3.76%)  | 163 (38.26%) |