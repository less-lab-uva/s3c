# Scripts for working with CARLA data
This folder contains scripts for working with the CARLA data from the study and is organized as follows:
* `avg_graph_size.py`
  * Computes the average size of the representative graph over the equivalence classes of a dataset. This script was used with `study_data/carla_clusters/carla_rsv.json` and `study_data/carla_clusters/carla_abstract.json` to find that the *ELR* abstraction on average used 16 nodes and 47 edges compared to 131 nodes and 264 edges for *ERS*.
* `cluster_figure_generator.py`
  * Parses the JSON files from `study_data/carla_clusters/` to generate figures showing the distribution of the equivalence class sizes as shown in Figure 3.
* `gen_clusters.sh`
  * Given the full study data, which will be provided as a separate download due to file size limits, this will generate the JSON files found in `study_data/carla_clusters/`
* `meta_figure_generator.py`
  * Consumes the data in `study_data/results` generated by `parse_clusters_carla.py` to generate Figure 4 and its relevant statistics.
* `parse_all_clusters.sh`
  * Shell script to repeatedly invoke `parse_clusters_carla.py` for the different abstractions.
* `parse_clusters_carla.py`
  * Parses the `study_data/model_results/0.8_percent_e386_results.csv` file generated by `rq1/new_study.py` into intermediate forms used to answer RQ1 and RQ2.