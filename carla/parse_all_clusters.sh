cd $(dirname ${BASH_SOURCE})/..
source env.sh
export OUTER_SAVE_FILE=study_data/results/
export CLUSTER_DIR=study_data/carla_clusters/
export PHYSCOV_CLUSTER_DIR=study_data/physcov_clusters/
export MODEL_RESULTS_DIR=study_data/model_results/
for results in "0.8_percent_e386"
do
  for dataset in "carla_abstract"
  do
    save_dir="${OUTER_SAVE_FILE}${results}/${dataset}/"
    echo "${save_dir}"
    mkdir -p "${save_dir}"
    python3 carla/parse_clusters_carla.py -dsf ${CLUSTER_DIR}/${dataset}.json -test "${MODEL_RESULTS_DIR}/${results}_results.csv" -o "${save_dir}"
  done
  for dataset in "carla_rsv"
  do
    save_dir="${OUTER_SAVE_FILE}${results}/${dataset}/"
    echo "${save_dir}"
    mkdir -p "${save_dir}"
    python3 carla/parse_clusters_carla.py -dsf ${CLUSTER_DIR}/${dataset}.json -test "${MODEL_RESULTS_DIR}/${results}_results.csv" -o "${save_dir}"
  done
  for dataset in "carla_sem_rel"
  do
    save_dir="${OUTER_SAVE_FILE}${results}/${dataset}/"
    echo "${save_dir}"
    mkdir -p "${save_dir}"
    python3 carla/parse_clusters_carla.py -dsf ${CLUSTER_DIR}/${dataset}.json -test "${MODEL_RESULTS_DIR}/${results}_results.csv" -o "${save_dir}"
  done
  for dataset in "carla_sem" "carla_no_rel"
  do
    save_dir="${OUTER_SAVE_FILE}${results}/${dataset}/"
    echo "${save_dir}"
    mkdir -p "${save_dir}"
    python3 carla/parse_clusters_carla.py -dsf ${CLUSTER_DIR}/${dataset}.json -test "${MODEL_RESULTS_DIR}/${results}_results.csv" -o "${save_dir}"
  done
  for dataset in "rss_10"
  do
    save_dir="${OUTER_SAVE_FILE}${results}/${dataset}/"
    echo "${save_dir}"
    mkdir -p "${save_dir}"
    python3 carla/parse_clusters_carla.py -dsf ${PHYSCOV_CLUSTER_DIR}/${dataset}.json -test "${MODEL_RESULTS_DIR}/${results}_results.csv" -o "${save_dir}"
  done
  for dataset in "rss_v2_10"
  do
    save_dir="${OUTER_SAVE_FILE}${results}/${dataset}/"
    echo "${save_dir}"
    mkdir -p "${save_dir}"
    python3 carla/parse_clusters_carla.py -dsf ${PHYSCOV_CLUSTER_DIR}/${dataset}.json -test "${MODEL_RESULTS_DIR}/${results}_results.csv" -o "${save_dir}"
  done
  for dataset in "carla_rsv_time_2" "carla_rsv_time_3" "carla_rsv_time_5" "carla_rsv_time_10" "carla_abstract_time_2" "carla_abstract_time_3" "carla_abstract_time_5" "carla_abstract_time_10"
  do
    save_dir="${OUTER_SAVE_FILE}${results}/${dataset}/"
    echo "${save_dir}"
    mkdir -p "${save_dir}"
    python3 carla/parse_clusters_carla.py -dsf ${CLUSTER_DIR}/${dataset}.json -test "${MODEL_RESULTS_DIR}/${results}_results.csv" -o "${save_dir}"
  done
  for dataset in "rss_v2_10_time_2" "rss_v2_10_time_3" "rss_v2_10_time_5" "rss_v2_10_time_10"
  do
    save_dir="${OUTER_SAVE_FILE}${results}/${dataset}/"
    echo "${save_dir}"
    mkdir -p "${save_dir}"
    python3 carla/parse_clusters_carla.py -dsf ${PHYSCOV_CLUSTER_DIR}/${dataset}.json -test "${MODEL_RESULTS_DIR}/${results}_results.csv" -o "${save_dir}"
  done
done

wait

echo "All jobs finished"
