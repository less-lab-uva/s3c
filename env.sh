#! /bin/bash

cd $(dirname ${BASH_SOURCE})
export S3C_BASE=$(pwd)
if conda activate ./sg 2> /dev/null ; then
    echo "Conda environment already created."
else
    echo "Creating conda environment."
    conda create --prefix=sg python=3.9 --yes
    conda activate ./sg
    python3 -m pip install -r requirements.txt
fi
export PYTHONPATH=$PYTHONPATH:$(pwd)
