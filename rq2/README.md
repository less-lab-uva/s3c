# Scripts for generating and parsing data for RQ2
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