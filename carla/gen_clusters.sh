cd $(dirname ${BASH_SOURCE})/..
conda activate sg
export PYTHONPATH=$(pwd)
export JOBS=30  # TODO tune for your system
export DATADIR=$TODO  # TODO: set this to the directory containing to be referenced by the dataloader
for i in "CarlaRSV carla_rsv"\
 "CarlaAbstract carla_abstract"\
 "CarlaSem carla_sem"\
 "CarlaSemRel carla_sem_rel"\
 "CarlaNoRel carla_no_rel"
do
  set -- $i # convert the "tuple" into the param args $1 $2...
  echo "Running ${1}"
  python3 rq3/rq3_runner.py -dt $1 -dp $DATADIR -sp $DATADIR -dsf ${DATADIR}/${2}.json -j $JOBS --verbose
done
