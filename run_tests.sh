#!/usr/bin/env bash
set -e
SDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ "$#" -eq 0 ]; then
    echo "Unittest discovery mode"
    echo "========================================"
    PYTHONPATH=. python -m unittest discover -s "${SDIR}/dosna/tests" -v
else
    echo "Unittest manual mode:" $@
    for script in 'connection' 'dataset' 'dataset_mpi';
    do
        echo "========================================"
        echo "Running tests on ${script}"
        echo "========================================"
        PYTHONPATH="${SDIR}:$PYTHONPATH"
        python "${SDIR}/dosna/tests/test_${script}.py" $@
    done
fi
