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
python $SCRIPT_DIR/expand.py --from_case case118 -n 1000 -s $1/train  >/dev/null

echo "Generating Validation Set"
python $SCRIPT_DIR/expand.py --from_case case118 -n 1000 -s $1/val  >/dev/null

echo "Generating Testing Set"
python $SCRIPT_DIR/expand.py --from_case case118 -n 1000 -s $1/test  >/dev/null
