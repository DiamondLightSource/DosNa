#!/usr/bin/env bash

if [ "$#" -eq 0 ]; then
    echo "Unittest discovery mode"
    echo "========================================"
    PYTHONPATH=. python -m unittest discover -s tests -v
else
    echo "Unittest manual mode:" $@
    for script in 'cluster' 'pool' 'dataset';
    do
        echo "========================================"
        echo "Running tests on ${script}"
        echo "========================================"
        PYTHONPATH=. python tests/test_${script}.py $@
    done
fi