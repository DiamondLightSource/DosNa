#!/usr/bin/env bash

if [ "$#" -eq 0 ]; then
    echo "Unittest discovery mode"
    echo "========================================"
    PYTHONPATH=. python -m unittest discover -s dosna/tests -v
else
    echo "Unittest manual mode:" $@
    for script in 'connection' 'dataset';
    do
        echo "========================================"
        echo "Running tests on ${script}"
        echo "========================================"
        PYTHONPATH=. python dosna/tests/test_${script}.py $@
    done
fi