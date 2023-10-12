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
echo "Generating from case1888rte"
python $SCRIPT_DIR/generate.py case1888rte -n 1000 -s $1/train --subgraphing_method num_change  >/dev/null


echo "Generating Val Set"
echo "Generating from case1888rte"
python $SCRIPT_DIR/generate.py case1888rte -n 100 -s $1/val --subgraphing_method num_change  >/dev/null
# echo "Generating from case30"
# python $SCRIPT_DIR/generate.py case30 -n 15 -s $1/train  >/dev/null
# echo "Generating from case_ieee30"
# python $SCRIPT_DIR/generate.py case_ieee30 -n 15 -s $1/train  >/dev/null
# echo "Generating from case39"
# python $SCRIPT_DIR/generate.py case39 -n 15 -s $1/train  >/dev/null
# echo "Generating from case57"
# python $SCRIPT_DIR/generate.py case57 -n 15 -s $1/train  >/dev/null
# echo "Generatingfrom case89pegase"
# python $SCRIPT_DIR/generate.py case89pegase -n 15 -s $1/train  >/dev/null
# echo "Generating from case118"
# python $SCRIPT_DIR/generate.py case118 -n 15 -s $1/train  >/dev/null
# echo "Generating from case145"
# python $SCRIPT_DIR/generate.py case145 -n 15 -s $1/train  >/dev/null
# echo "Generating from case_illinois200"
# python $SCRIPT_DIR/generate.py case_illinois200 -n 15 -s $1/train  >/dev/null
# echo "Generating from case300"
# python $SCRIPT_DIR/generate.py case300 -n 15 -s $1/train  >/dev/null
# echo "Generating from case1354pegase"
# python $SCRIPT_DIR/generate.py case1354pegase -n 15 -s $1/train  >/dev/null
# echo "Generating from case1888rte"
# python $SCRIPT_DIR/generate.py case1888rte -n 15 -s $1/train  >/dev/null
# echo "Generating from case2848rte"
# python $SCRIPT_DIR/generate.py case2848rte -n 15 -s $1/train  >/dev/null
# echo "Generating from case2869pegase"
# python $SCRIPT_DIR/generate.py case2869pegase -n 15 -s$1/train  >/dev/null

# echo "Generating Validation Set"
# python $SCRIPT_DIR/generate.py GBnetwork -n 100 -max_size 200 -s $1/val  >/dev/null

# echo "Generating Testing Set"
# python $SCRIPT_DIR/generate.py GBnetwork -n 100 -s $1/test  >/dev/null
