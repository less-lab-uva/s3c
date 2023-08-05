# Precomputed Study Data and Figure Scripts
This folder contains precomputed study data and scripts for generating the figures.
The folder is organized as follows:
* `carla_clusters/`
  * Contains the JSON files of the clusterings for each of the 5 abstractions explored as part of the study. For more information on the naming scheme, see `s3c/README.md`.
* `figures/`
  * This folder will be automatically generated with `generate_figures.sh` and contain Figures 3 and 4, as well as different variations on those figures.
* `model_results/0.8_percent_e386_results.csv`
  * Contains the predictions of the model trained in RQ1
* `results/`
  * This folder will be automatically generated with `generate_figures.sh` and contains intermediate data needed to compute the data for RQ1 and RQ2.
* `generate_figures.sh`
  * Script for generating all relevant figures and tables. Run as `source study_data/generate_figures.sh`.