#! /bin/bash

cd $(dirname ${BASH_SOURCE})/..
echo "Parsing PhysCov comparison data for RQ1"
source carla/parse_physcov.sh
echo "Generating time sequence data"
source carla/gen_time_sequence.sh
echo "Parsing data for RQ1 and RQ2"
source carla/parse_all_clusters.sh
echo "Generating figures for Approach"
python3 carla/cluster_figure_generator.py -i study_data/carla_clusters/ -o study_data/figures/
echo "Generating figures for RQ1"
python3 carla/meta_figure_generator.py -i study_data/results/ -o study_data/figures/
echo "Generating figures for Approach based on time data"
python3 carla/meta_figure_generator.py -i study_data/results/ -o study_data/figures/ --time
echo "Generating RQ2-A data"
python3 rq2/a/rq2a.py
echo "Generating RQ2-B data"
python3 rq2/b/rq2b.py
cd exploratory_work
source unpack_exploratory_work.sh
python3 exploratory_work.py
cd ../