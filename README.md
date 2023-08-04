<h1>S<sup>3</sup>C: Spatial Semantic Scene Coverage for Autonomous Vehicles</h1>
This repository contains code and scripts to reproduce the results from S<sup>3</sup>C: Spatial Semantic Scene Coverage for Autonomous Vehicles.

# Requirements
The system uses anaconda and Python 3.9, which must be installed on the host system.
Running `source env.sh` (which is done automatically by the below scripts), will set up a conda environment called `sg` and install all Python requirements.

# Generating Figures and Data
The `study_data` contains scripts for generating the figures and tables in the paper.
The `study_data/carla_clusters/` folder contains the precomputed clusterings of the 5 techniques explored in Section 4.2.
The `study_data/model_results/` folder contains a CSV file with the predictions of the trained model for the entire train and test splits of the data.

To recreate the figures and data, run:

```bash
source study_data/generate_figures.sh
```

## Figure information
| Paper Figure | File | Description                                                                                                       |
|--------------|---|-------------------------------------------------------------------------------------------------------------------|
| Fig. 3       | `study_data/figures/cluster_viz_carla_rsv.png`  | Distribution of images across scene graph equivalence classes for the *ELR* abstaction.                           |
| Fig. 4       | `study_data/figures/num_clusters_80_20_trivial_inner_legend.png`  | Percentage of test failures not covered in training vs count of equivalence classes under different abstractions. |
| Fig. 5       | `study_data/figures/tree.png`  | Test fail and Train classification tree for the *ELR* abstraction. |

### File naming scheme
In the paper, we introduce 5 abstractions, *E*, *EL*, *ER*, *ELR*, and *ERS*.
The short names used in the file names are different. The various figures and supporting data use the below naming scheme.

| Short Name | Long Name                    | Description                                                               | File Ending |
|------------|------------------------------|---------------------------------------------------------------------------|-------------|
| *E*        | Entities                     | Semantic Segmentation                                                     | `_sem`      |
| *EL*       | Entities + Lanes             | *E* with ground-truth lane occupation for each entity                     | `_no_rel`   |
| *ER*       | Entities + Relations         | *E* with RoadScene2Vec's default inter-entity relationships configuration | `_sem_rel`  |
| *ELR*      | Entities + Lanes + Relations | *E* with both lane and relationship information from *EL* and *ER*        | `_rsv`      |
| *ERS*      | Entities + Road Structure    | *EL* except lanes are modeled as multiple detailed road segments          | `_abstract` |
