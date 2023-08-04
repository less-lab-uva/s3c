#! /bin/bash

cd $(dirname ${BASH_SOURCE})
if conda activate sg 2> /dev/null ; then
    echo "Conda environment already created."
else
    echo "Creating conda environment."
    conda create --prefix=sg python=3.9 --yes
    pip install -r requirements.txt
    conda activate sg 2> /dev/null
fi
export PYTHONPATH=$PYTHONPATH:$(pwd)