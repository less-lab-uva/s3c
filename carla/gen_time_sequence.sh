cd $(dirname ${BASH_SOURCE})/..
source env.sh
if [ -z "$1" ]
  then
    # if no args are passed, default to the expected ones for general replication
    export CARLA_CLUSTER_DIR=study_data/carla_clusters/
    export PHYSCOV_CLUSTER_DIR=study_data/physcov_clusters/
else
    export CARLA_CLUSTER_DIR=$1
    export PHYSCOV_CLUSTER_DIR=$2
fi

for time in "2" "3" "5" "10"
do
  for dataset in "carla_rsv" "carla_abstract"
  do
    python3 carla/gen_time_sequence.py -time ${time} -dataset "${CARLA_CLUSTER_DIR}/${dataset}.json"  -output ${CARLA_CLUSTER_DIR}
  done
done

for time in "2" "3" "5" "10"
do
  for dataset in "rss_v2_10"
  do
    python3 carla/gen_time_sequence.py -time ${time} -dataset "${PHYSCOV_CLUSTER_DIR}/${dataset}.json"  -output ${PHYSCOV_CLUSTER_DIR}
  done
done
