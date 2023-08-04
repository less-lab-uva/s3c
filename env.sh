#! /bin/bash

cd $(dirname ${BASH_SOURCE})
if conda activate sg 2> /dev/null ; then
    echo "Conda environment already created."
else
    echo "Creating conda environment."
    conda create --prefix=sg python=3.9 --yes
    conda activate sg 2> /dev/null
    pip install -r requirements.txt
fi
export PYTHONPATH=$PYTHONPATH:$(pwd)