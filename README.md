<p align="center">
  <img src="https://d2k0ddhflgrk1i.cloudfront.net/Websections/Huisstijl/Bouwstenen/Logo/02-Visual-Bouwstenen-Logo-Varianten-v1.png"/><br>
  <br><br>
</p>

## Authors

Francesco Piccoli

Àlex De Los Santos Subirats

Marcus Plesner

Marco van Veen

# Inductive GNNs for power grids
Continuous changes in the topology of power grids due to factors such as faults, the integration of renewable energy sources, and system expansions make it increasingly challenging to promptly mon- itor the state, evaluate conditions, and effectively address the issues that can arise within a power system. In this dynamic environment, methods capable of rapidly inferring the grid’s state with accuracy and adaptability are of crucial importance. In our project, we leverage Graph Neural Networks (GNNs) to devise an inductive framework that is applicable to common power systems operational calcualations, such as power flow and optimal power flow. The framework aims to provide fast and accurate predictions despite evolving topology of the power grid.


## Codebase setup
```
**macOS/Linux**
git clone https://github.com/AlexDeLos/GNN_OPF.git
cd GNN_OPF
virtualenv gnn_opf_env -p python3.10.8 
source gnn_opf_env/bin/activate
pip3 install -r requirements.txt
```

## Repository tree
```
├── data_generation
│   ├── expand.py
│   ├── generate.py
│   ├── generate_inductive_dataset.sh
│   ├── generate_transductive_physics_dataset.sh
│   └── subgraphs_methods.py
└── src
    ├── hyperparams_optimization.py
    ├── main.py
    ├── test.py
    ├── test_physics.py
    ├── train
    ├── models
    │   ├── GAT.py
    │   ├── GINE.py
    │   ├── GraphSAGE.py
    │   ├── HeterogenousGNN.py
    │   ├── MessagePassing.py
    └── utils
        ├── utils.py
        ├── utils_hetero.py
        ├── utils_homo.py
        ├── utils_physics.py
        └── utils_plot.py
```

## Generate dataset

#### Inductive dataset
By default it will generate 15 subgraphs for each power grid listed [here](https://pandapower.readthedocs.io/en/v2.4.0/networks/power_system_test_cases.html). 
The default subgraphing technique is `rnd_neighbor`, refer to `data_generation/generate.py` for all the possible customizable arguments and their default values.
```
bash data_generation/generate_inductive_dataset.sh ./Data
```


#### Transductive dataset

By default it will take [case118](https://pandapower.readthedocs.io/en/v2.4.0/networks/power_system_test_cases.html#case-118) power grid and create syntethic data (modifying node features) while maintaing the same power grid topology. It will create 3 datasets: `train`, `val`, `test`, each containing 1000 instances. Refer to `data_generation/expand.py` for all the possible customizable arguments and their default values. It is also possible to create new syntethic instances for all graphs inside a directory.  
```
bash data_generation/generate_inductive_dataset.sh ./Data
```

## Train a model

### Train
The first time the model is trained, it will create the data instances from the raw data and save them to pickle files inside the dataset folders. The data instances created in pickle files will only be compatible for heterogenous graphs tranining (or viceversa). To run homogenous graph models new data instances need to be created by removing or moving the heterogenous pickle files from the dataset directories (or viceversa).
Check src/utils/utils.py for all possible arguments and training settings. The default parameter arguments for the different architectures are not currently used as they differ between architectures and each architecture performs best with different parameters.

#### Heterogenous graph model
To train a heterogenous model run the following command. To customize the heterogenous convolution used (`GAT`, `GATv2`, `GINE`, `SAGE`), as well all the other model hyperparameters, modify the `__init__` arguments of `HeteroGNN` class inside `src/models/HeterogenousGNN` file. 
```
python src/main.py HeteroGNN
```

#### Homogenous graph model
To train a homogenous model run the following command, other options in addition to `GAT` are (`GINE`, `GraphSAGE`, `MessagePassing`).
```
python src/main.py GAT
```

### Loss
It is possible to train the model using 3 different loss functions: `standard`, `physics`, and `mixed`.
* `standard` refers to a supervised training setting, where the model is trained to imitate the output of a numerical solver, the loss used is specified in the `criterion` argument, default is `MSELoss`.
* `physics` trains in an unsupervised manner, using a physics-informed loss,  by training to minimizing the power imbalances in the power flow equations.
* `mixed` is a combination of the previous two, `mixed_loss_weight` argument specifies the weight to give to the `standard` loss, default is `0.1`.

`standard` and `physics` require the use of different features for the nodes, therefore the data instances created for one loss, and saved to the pickle files in the dataset folders, cannot be reused for the other loss. As for homogenous/heterogenous new data instances need to be created by moving the pickle files from the dataset directories.



## Test
The trained model parameters will be saved in a `./trained_models` folder, every 10 epochs.
To test the performance of a model it is possible to run:
```
python src/test.py -g <model_type> -m <trained_model_path.pt> -d <test_dataset_directory>
```
For instance:
```
python src/test.py -g HeteroGNN -m ./trained_models/D5B723AI-HeteroGNN/D5B723AI-HeteroGNN_final.pt -d ./Data/test
```
To test model trained with a `physics` loss or a `mixed` loss, use `test_physics.py` instead of `test.py` file.



## Results

