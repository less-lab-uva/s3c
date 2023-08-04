#! /bin/bash

cd $(dirname ${BASH_SOURCE})/..
echo "Parsing data for RQ1 and RQ2"
source carla/parse_all_clusters.sh
echo "Generating figures for Approach"
python3 carla/cluster_figure_generator.py -i study_data/carla_clusters/ -o study_data/figures/
echo "Generating figures for RQ1"
python3 carla/meta_figure_generator.py -i study_data/results/ -o study_data/figures/
echo "Generating RQ2-A data"
python3 rq2/a/rq2a.py
echo "Generating RQ2-B data"
python3 rq2/b/rq2b.py