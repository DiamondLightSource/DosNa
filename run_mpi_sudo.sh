#!/bin/sh

MPI_INC=$(readlink -f "$(dirname `which mpirun`)/../lib")

sudo LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MPI_INC PYTHONPATH=$PYTHONPATH:. mpiexec -n $1 python ${@:2}
