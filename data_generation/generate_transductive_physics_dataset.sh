#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [ $# -eq 0 ]
  then
    echo "Supply data saving path"
    exit
fi

echo "running script in " $SCRIPT_DIR
echo "saving in " $1

echo "Generating Training Set"
echo "Generating from case118"
C:\\Users\\Marco\\Documents\\GNN_OPF\\venv\\Scripts\\python.exe $SCRIPT_DIR/generate.py case118 --subgraphing_method num_change --min_size 10 --no_leakage -n 10000 -s $1/train  >/dev/null

echo "Generating Validation Set"
C:\\Users\\Marco\\Documents\\GNN_OPF\\venv\\Scripts\\python.exe $SCRIPT_DIR/generate.py case118 --subgraphing_method num_change --no_leakage -n 1000 -s $1/val  >/dev/null

echo "Generating Testing Set"
C:\\Users\\Marco\\Documents\\GNN_OPF\\venv\\Scripts\\python.exe $SCRIPT_DIR/generate.py case118 --subgraphing_method num_change --no_leakage -n 1000 -s $1/test  >/dev/null
