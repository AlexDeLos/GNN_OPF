#!/bin/bash

# Naming convention:
#
# Format predictive_strength-value_mode-normalized-linear-model-loss_mode-model
# predictive_strength: [Transductive, Inductive] (T/I)
# value_mode: [All, Missing, Two Voltage] (A/M/V)
# normalized: [Normalized, Un-normalized] (N/U)
# linear: [Linear, No-Linear] (L/N)
# loss_mode: [Supervised, Unsupervised, Mix] (S/U/M)
# model: [GAT, GraphSAGE] (automatically in name)


predictive_strength=("transductive" "inductive") # Data selection
value_mode=("all" "missing" "voltage") 
normalized=("unnormalized" "normalized") # (Binary)
linear=("no_linear" "linear") # (Binary)
loss_mode=("supervised" "unsupervised" "mix")
model=("GAT" "GraphSAGE")

counter=0

for pred in "${predictive_strength[@]}"; do
    for val in "${value_mode[@]}"; do
        for norm in "${normalized[@]}"; do
            for lin in "${linear[@]}"; do
                for mod in "${model[@]}"; do
                    if [ "$norm" = "normalized" ]; then
                        norm_flag="--normalize"
                    else
                        norm_flag=""
                    fi

                    if [ "$lin" = "no_linear" ]; then
                        lin_flag="--no_linear"
                    else
                        lin_flag=""
                    fi
                    echo "Current Experiment: $counter"
                    echo "Running $pred experiment, predicting $val values, with $norm features, using the $mod model, in $lin mode"
                    # python src/main.py "$mod" --train "./Data/testing/$pred/$val" --val "./Data/testing/test_set/$val" --test "./Data/testing/test_set" $norm_flag $lin_flag --loss_mode "$loss" -m "$pred-$val-$norm-$lin-$loss"
                    # python src/test.py -g "$mod" -m "./trained_models/$pred-$val-$norm-$lin-$loss-$mod.pt" -d "./Data/testing/$pred/$val"
                    ((counter++))
                    sleep 1
                done
            done
        done
    done
done
echo "Total Experiments: $counter"

        # for use_flag in true false; do
        #     if [ "$use_flag" = true ]; then
        #         flag="--store_true_flag"
        #     else
        #         flag=""
        #     fi

        #     # Execute Python script with different parameters
        #     python3 your_python_script.py --param1 "$param1" --param2 "$param2" $flag

        #     # Optional: add a sleep to avoid overloading the system
        #     sleep 1
        # done