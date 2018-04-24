#!/bin/bash
make
make clovis-test
LD_LIBRARY_PATH="$(pwd)"
cp -f clovis-test pyclovis
./pyclovis
