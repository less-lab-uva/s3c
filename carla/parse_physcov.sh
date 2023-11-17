cd $(dirname ${BASH_SOURCE})/..
source env.sh
export PHYSCOV_DIR=study_data/physcov/
export CARLA_CLUSTER_DIR=study_data/carla_clusters/
export CLUSTER_DIR=study_data/physcov_clusters/
python3 carla/parse_physcov.py -input ${PHYSCOV_DIR} -output ${CLUSTER_DIR}