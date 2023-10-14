
#!/bin/bash

values1=(0 1 2)
values2=(8 16 32 64 128)
values3=(0 1 2)
values4=(8 16 32 64 128)

echo "GraphSAGE tests"

for i in "${values1[@]}"; do
    for j in "${values2[@]}"; do
        for k in "${values3[@]}"; do
            for l in "${values4[@]}"; do
                echo "gnn layers=$i, gnn dim=$j, lin layers=$k, lin dim=$l"
                python src/main.py GraphSAGE --train ./Data/rnd_gen/large --val ./Data/test118 -m "$i-$j-$k-$l" --n_hidden_gnn "$i" --gnn_hidden_dim "$j"  --n_hidden_lin "$k"  --lin_hidden_dim "$l" 
                python src/test.py -g GraphSAGE -m "./trained_models/$i-$j-$k-$l-GraphSAGE.pt" -d ./Data/test118  --n_hidden_gnn "$i" --gnn_hidden_dim "$j"  --n_hidden_lin "$k"  --lin_hidden_dim "$l" 
            done
        done
    done
done

echo "GAT tests"

for i in "${values1[@]}"; do
    for j in "${values2[@]}"; do
        for k in "${values3[@]}"; do
            for l in "${values4[@]}"; do
                echo "gnn layers=$i, gnn dim=$j, lin layers=$k, lin dim=$l"
                python src/main.py GAT --train ./Data/rnd_gen/large --val ./Data/test118 -m "$i-$j-$k-$l" --n_hidden_gnn "$i" --gnn_hidden_dim "$j"  --n_hidden_lin "$k"  --lin_hidden_dim "$l" 
                python src/test.py -g GAT -m "./trained_models/$i-$j-$k-$l-GAT.pt" -d ./Data/test118 --n_hidden_gnn "$i" --gnn_hidden_dim "$j"  --n_hidden_lin "$k"  --lin_hidden_dim "$l" 
            done
        done
    done
done

echo "GINE tests"

for i in "${values2[@]}"; do
    for j in "${values4[@]}"; do
        echo "gnn dim=$i, gnn dim=$j"
        python src/main.py GINE --train ./Data/rnd_gen/large --val ./Data/test118 -m "$i-$j" --n_hidden_gnn 0 --gnn_hidden_dim "$i" --n_hidden_lin 0 --lin_hidden_dim "$j" 
        python src/test.py -g GINE -m "./trained_models/$i-$j-GINE.pt" -d ./Data/test118 --n_hidden_gnn "$i" --gnn_hidden_dim "$j"  --n_hidden_lin "$k"  --lin_hidden_dim "$l" 
    done
done
