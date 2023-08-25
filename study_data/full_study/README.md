# Full Study Recreation
As opposed to the files in the `study_data` folder, e.g. `generate_figures.sh` that use the intermediate data to generate the figures and data, this folder contains scripts to download the raw data from the study and run the full analysis scripts.

The `full_study.sh` script will download ~63 GB compressed from Zenodo and extract it into 8 folders in full_study_data/; the data is 66 GB uncompressed. 
This download will only happen once.
The full download, extraction, and analysis will take 2-3 hours with high variability based on your network connection and computing power.

***The system will attempt to run all analysis fully parallelized after the download completes - this may lead to the system lagging as the script uses extensive system resources.***

To run the script:
```bash
# from the root folder:
source env.sh
source study_data/full_study/full_study.sh
# the system will run for 2-3 hours, printing progress as it runs.
```

## Files Downloaded
The system will download the following raw data

* `full_study/full_study_data/`
  * Graphs and images from the 8 different experiment map and parameter combinations:
    * 4.7G    `./Town01_zero_car`
    * 7.4G    `./Town01_max_car`
    * 4.0G    `./Town02_zero_car`
    * 5.0G    `./Town02_max_car`
    * 14G     `./Town04_zero_car`
    * 20G     `./Town04_max_car`
    * 5.5G    `./Town10HD_zero_car`
    * 7.2G    `./Town10HD_max_car`

## Files Generated
* `full_study/carla_clusters/`
    * Equivalent clusters to those found in `study_data/carla_clusters`. Note that the file contents might differ due to ordering differences, though semantics are the same. Running `full_study.sh` will use a short Python script to load the clusters and check semantic equality of the Python clustering objects.
* `full_study/results/`
    * Equivalent intermediate data to that calculated by `generate_figures.sh` and stored in `study_data/results/`. Note that the file contents might be different due to ordering differences, though semantics are the same.
* `full_study/figures/`
    * Equivalent figures to those created by `generate_figures.sh` and found in `study_data/figures`.
* `rq2/a/splits_csv_orig/`
    * CSV files containing the scene graphs that belong to the test failure and the train split.
* `rq2/b/carla_csv_orig/`
    * CSV files containing the scene graphs that belong to each of the towns with zero and max cars.