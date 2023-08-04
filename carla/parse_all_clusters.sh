cd $(dirname ${BASH_SOURCE})/..
source env.sh
export OUTER_SAVE_FILE=study_data/results/
export CLUSTER_DIR=study_data/carla_clusters/
export MODEL_RESULTS_DIR=study_data/model_results/
for results in "0.8_percent_e386"
do
  for dataset in "carla_abstract" "carla_rsv" "carla_sem" "carla_sem_rel" "carla_no_rel"
  do
    save_dir="${OUTER_SAVE_FILE}${results}/${dataset}/"
    echo "${save_dir}"
    mkdir -p "${save_dir}"
    python3 carla/parse_clusters_carla.py -dsf ${CLUSTER_DIR}/${dataset}.json -test "${MODEL_RESULTS_DIR}/${results}_results.csv" -o "${save_dir}"
  done
done

wait

echo "All jobs finished"
