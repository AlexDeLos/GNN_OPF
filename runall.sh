
echo "Starting RND Experiments"

python ./src/main.py GAT --train ./Data/rnd_gen/small --val ./Data/test118 -m small_rnd
python ./src/main.py GAT --train ./Data/rnd_gen/medium --val ./Data/test118 -m medium_rnd
python ./src/main.py GAT --train ./Data/rnd_gen/large --val ./Data/test118 -m large_rnd

python ./src/main.py GINE --train ./Data/rnd_gen/small --val ./Data/test118 -m small_rnd
python ./src/main.py GINE --train ./Data/rnd_gen/medium --val ./Data/test118 -m medium_rnd
python ./src/main.py GINE --train ./Data/rnd_gen/large --val ./Data/test118 -m large_rnd

python ./src/main.py GraphSAGE --train ./Data/rnd_gen/small --val ./Data/test118 -m small_rnd
python ./src/main.py GraphSAGE --train ./Data/rnd_gen/medium --val ./Data/test118 -m medium_rnd
python ./src/main.py GraphSAGE --train ./Data/rnd_gen/large --val ./Data/test118 -m large_rnd

python ./src/main.py MessagePassing --train ./Data/rnd_gen/small --val ./Data/test118 -m small_rnd
python ./src/main.py MessagePassing --train ./Data/rnd_gen/medium --val ./Data/test118 -m medium_rnd
python ./src/main.py MessagePassing --train ./Data/rnd_gen/large --val ./Data/test118 -m large_rnd

echo "Starting BFS experiments"

python ./src/main.py GAT --train ./Data/bfs_gen/small --val ./Data/test118 -m small_bfs
python ./src/main.py GAT --train ./Data/bfs_gen/medium --val ./Data/test118 -m medium_bfs
python ./src/main.py GAT --train ./Data/bfs_gen/large --val ./Data/test118 -m large_bfs

python ./src/main.py GINE --train ./Data/bfs_gen/small --val ./Data/test118 -m small_bfs
python ./src/main.py GINE --train ./Data/bfs_gen/medium --val ./Data/test118 -m medium_bfs
python ./src/main.py GINE --train ./Data/bfs_gen/large --val ./Data/test118 -m large_bfs

python ./src/main.py GraphSAGE --train ./Data/bfs_gen/small --val ./Data/test118 -m small_bfs
python ./src/main.py GraphSAGE --train ./Data/bfs_gen/medium --val ./Data/test118 -m medium_bfs
python ./src/main.py GraphSAGE --train ./Data/bfs_gen/large --val ./Data/test118 -m large_bfs

python ./src/main.py MessagePassing --train ./Data/bfs_gen/small --val ./Data/test118 -m small_bfs
python ./src/main.py MessagePassing --train ./Data/bfs_gen/medium --val ./Data/test118 -m medium_bfs
python ./src/main.py MessagePassing --train ./Data/bfs_gen/large --val ./Data/test118 -m large_bfs