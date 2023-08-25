export BASE_DIR=${S3C_BASE}/study_data/full_study/
cd ${BASE_DIR}
export DATADIR=${BASE_DIR}/full_study_data/
mkdir -p $DATADIR
export CLUSTER_DIR=${BASE_DIR}/carla_clusters/
export STUDY_CLUSTER_DATA_DIR=${S3C_BASE}/study_data/carla_clusters/
export FIGURE_DIR=${BASE_DIR}/figures/
export STUDY_FIGURE_DIR=${S3C_BASE}/study_data/figures/
pwd
mkdir -p ${CLUSTER_DIR}
if test -f ".zenodo" ; then
  printf "One time data download has already been completed, skipping.\n"
  printf "Previously downloaded data can be found in full_study_data/\n"
else
  printf "Downloading full study data from Zenodo.\n"
  printf "This will download and extract 8 folders in full_study_data/.\n"
  printf "The data totals 63GB compressed and 76GB uncompressed.\n"
  printf "This will take a very long time (~1.5 hours in testing with high variability).\n"
  printf "Started at $(date)\n"
  printf "Downloading Town01_zero_car (4.3 GB)\n" && \
  curl https://zenodo.org/record/8271428/files/Town01_zero_car.tar.gz?download=1 | tar xz -C full_study_data && \
  printf "Downloading Town02_zero_car (3.7 GB)\n" && \
  curl https://zenodo.org/record/8271428/files/Town02_zero_car.tar.gz?download=1 | tar xz -C full_study_data && \
  printf "Downloading Town04_zero_car (12.0 GB)\n" && \
  curl https://zenodo.org/record/8271428/files/Town04_zero_car.tar.gz?download=1 | tar xz -C full_study_data && \
  printf "Downloading Town10HD_zero_car (5.2 GB)\n" && \
  curl https://zenodo.org/record/8271428/files/Town10HD_zero_car.tar.gz?download=1 | tar xz -C full_study_data && \
  printf "Downloading Town01_max_car (6.8 GB)\n" && \
  curl https://zenodo.org/record/8250740/files/Town01_max_car.tar.gz?download=1 | tar xz -C full_study_data && \
  printf "Downloading Town02_max_car (4.7 GB)\n" && \
  curl https://zenodo.org/record/8250740/files/Town02_max_car.tar.gz?download=1 | tar xz -C full_study_data && \
  printf "Downloading Town04_max_car (17.8 GB)\n" && \
  curl https://zenodo.org/record/8250740/files/Town04_max_car.tar.gz?download=1 | tar xz -C full_study_data && \
  printf "Downloading Town10HD_max_car (6.7 GB)\n" && \
  curl https://zenodo.org/record/8250740/files/Town10HD_max_car.tar.gz?download=1 | tar xz -C full_study_data && \
  touch .zenodo && printf "Finished at $(date)\n"
fi
if test -f ".zenodo" ; then
  printf "Generating the clusterings and saving to carla_clusters/\n"
  printf "The generated clusters will be equivalent to those found in s3c/study_data/carla_clusters\n"
  cd ${S3C_BASE}
  pwd
  source env.sh
  export JOBS=$(python3 -c "import multiprocessing; print(max(multiprocessing.cpu_count()-2, 1))")
  for i in "CarlaRSV carla_rsv"\
   "CarlaAbstract carla_abstract"\
   "CarlaSem carla_sem"\
   "CarlaSemRel carla_sem_rel"\
   "CarlaNoRel carla_no_rel"
  do
    set -- $i # convert the "tuple" into the param args $1 $2...
    echo "Generating for ${1}"
    echo $DATADIR
    python3 utils/cluster_generator.py -dt $1 -dp $DATADIR -sp $DATADIR -dsf ${CLUSTER_DIR}/${2}.json -j $JOBS --verbose
  done
  for dataset in "carla_abstract" "carla_rsv" "carla_sem" "carla_sem_rel" "carla_no_rel"
  do
    echo "Checking that ${dataset}.json matches between what was calculated locally and what was provided before."
    python3 -c "from utils.dataset import Dataset; exit(0 if Dataset.load_from_file('${CLUSTER_DIR}/${dataset}.json') == Dataset.load_from_file('${STUDY_CLUSTER_DATA_DIR}/${dataset}.json') else 1)"
    if [ $? -eq 0 ]; then
      echo "Clusterings match!"
    else
      echo "Failed to match the clusterings for ${dataset}.json"
      return
    fi
  done
  echo "Parsing data for RQ1 and RQ2 from generated clusters. Started at $(date)"
  export OUTER_SAVE_FILE=study_data/full_study/results/
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
  echo "Finished parsing data at $(date)"
  echo "Generating figures for Approach"
  python3 carla/cluster_figure_generator.py -i ${CLUSTER_DIR}/ -o ${FIGURE_DIR}
  echo "Generating figures for RQ1"
  python3 carla/meta_figure_generator.py -i ${OUTER_SAVE_FILE}/ -o ${FIGURE_DIR}

  echo "Creating CSV files for RQ2-A"
  python3 rq2/a/create_csv_for_rq2a.py
  echo "Creating CSV files for RQ2-B"
  python3 rq2/b/create_csv_for_rq2b.py
  echo "Generating RQ2-A data"
  python3 rq2/a/rq2a.py
  echo "Generating RQ2-B data"
  python3 rq2/b/rq2b.py
  
else
  printf "Error downloading data at $(date). Please try again.\n"
fi
