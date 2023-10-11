echo "Starting RND Testing"

echo "small GAT"
python ./src/test.py -g GAT -m ./trained_models/small_rnd_GAT.pt -d ./Data/test118   
echo "medium GAT"
python ./src/test.py -g GAT -m ./trained_models/medium_rnd_GAT.pt -d ./Data/test118  
echo "large GAT"
python ./src/test.py -g GAT -m ./trained_models/large_rnd_GAT.pt -d ./Data/test118   

echo "small GINE"
python ./src/test.py -g GINE -m ./trained_models/small_rnd_GINE.pt -d ./Data/test118   
echo "medium GINE"
python ./src/test.py -g GINE -m ./trained_models/medium_rnd_GINE.pt -d ./Data/test118  
echo "large GINE"
python ./src/test.py -g GINE -m ./trained_models/large_rnd_GINE.pt -d ./Data/test118   

echo "small GraphSAGE"
python ./src/test.py -g GraphSAGE -m ./trained_models/small_rnd_GraphSAGE.pt -d ./Data/test118   
echo "medium GraphSAGE"
python ./src/test.py -g GraphSAGE -m ./trained_models/medium_rnd_GraphSAGE.pt -d ./Data/test118  
echo "large GraphSAGE"
python ./src/test.py -g GraphSAGE -m ./trained_models/large_rnd_GraphSAGE.pt -d ./Data/test118   

echo "small MessagePassing"
python ./src/test.py -g MessagePassing -m ./trained_models/small_rnd_MessagePassing.pt -d ./Data/test118   
echo "medium MessagePassing"
python ./src/test.py -g MessagePassing -m ./trained_models/medium_rnd_MessagePassing.pt -d ./Data/test118  
echo "large MessagePassing"
python ./src/test.py -g MessagePassing -m ./trained_models/large_rnd_MessagePassing.pt -d ./Data/test118   

echo "Starting BFS Testing"

echo "small GAT"
python ./src/test.py -g GAT -m ./trained_models/small_bfs_GAT.pt -d ./Data/test118   
echo "medium GAT"
python ./src/test.py -g GAT -m ./trained_models/medium_bfs_GAT.pt -d ./Data/test118  
echo "large GAT"
python ./src/test.py -g GAT -m ./trained_models/large_bfs_GAT.pt -d ./Data/test118   

echo "small GINE"
python ./src/test.py -g GINE -m ./trained_models/small_bfs_GINE.pt -d ./Data/test118   
echo "medium GINE"
python ./src/test.py -g GINE -m ./trained_models/medium_bfs_GINE.pt -d ./Data/test118  
echo "large GINE"
python ./src/test.py -g GINE -m ./trained_models/large_bfs_GINE.pt -d ./Data/test118   

echo "small GraphSAGE"
python ./src/test.py -g GraphSAGE -m ./trained_models/small_bfs_GraphSAGE.pt -d ./Data/test118   
echo "medium GraphSAGE"
python ./src/test.py -g GraphSAGE -m ./trained_models/medium_bfs_GraphSAGE.pt -d ./Data/test118  
echo "large GraphSAGE"
python ./src/test.py -g GraphSAGE -m ./trained_models/large_bfs_GraphSAGE.pt -d ./Data/test118   

echo "small MessagePassing"
python ./src/test.py -g MessagePassing -m ./trained_models/small_bfs_MessagePassing.pt -d ./Data/test118   
echo "medium MessagePassing"
python ./src/test.py -g MessagePassing -m ./trained_models/medium_bfs_MessagePassing.pt -d ./Data/test118  
echo "large MessagePassing"
python ./src/test.py -g MessagePassing -m ./trained_models/large_bfs_MessagePassing.pt -d ./Data/test118   
