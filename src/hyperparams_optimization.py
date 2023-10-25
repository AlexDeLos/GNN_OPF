import optuna
import torch
from torch_geometric.loader import DataLoader as pyg_DataLoader
import tqdm
import os
import sys
# local imports
# add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.train_homo import train_batch, evaluate_batch
from train.train_hetero import train_batch_hetero, evaluate_batch_hetero
from utils.utils import get_gnn, load_data, get_criterion
from utils.utils_homo import normalize_data

import warnings
# Suppress FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# Hyperparameter optimization
def hyperparams_optimization(
        train,
        model_class_name="GraphSAGE", 
        n_trials=50, 
        learning_rate_range=(0.01, 0.1), # ranges for hyperparameters
        batch_size_values=[32, 64], 
        dropout_rate_range = (0.1, 0.3),
        num_epochs=200, 
        patience=30,
        optimizer_class=torch.optim.Adam,
        criterion_function="MSELoss",
    ):
    """
    Perform hyperparameter optimization for a given GNN model using Optuna library.

    Args:
        train (torch_geometric.data.Data): The training dataset.
        model_class_name (str): The name of the GNN model class to use. Default is "GraphSAGE".
        n_trials (int): The number of trials to run for hyperparameter optimization. Default is 50.
        learning_rate_range (tuple): The range of learning rates to search over. Default is (0.01, 0.1).
        batch_size_values (list): The list of batch sizes to search over. Default is [32, 64].
        dropout_rate_range (tuple): The range of dropout rates to search over. Default is (0.1, 0.3).
        num_epochs (int): The number of epochs to train for each trial. Default is 200.
        patience (int): The number of epochs to wait before early stopping if validation loss does not improve. Default is 30.
        optimizer_class (torch.optim.Optimizer): The optimizer class to use. Default is torch.optim.Adam.
        criterion_function (str): The name of the loss function to use. Default is "MSELoss".

    Returns:
        None
    """

    def objective(trial):

        input_dim = train[0].x.shape[1]
        edge_attr_dim = train[0].edge_attr.shape[1]
        output_dim = train[0].y.shape[1]

        print(f"Input shape: {input_dim}\nOutput shape: {output_dim}")

        batch_size = trial.suggest_categorical('batch_size', batch_size_values)
        
        train_dataloader = pyg_DataLoader(train, batch_size=batch_size, shuffle=True)
        val_dataloader = pyg_DataLoader(val, batch_size=batch_size, shuffle=False)

        gnn_class = get_gnn(model_class_name)

        dropout_rate = trial.suggest_float('dropout_rate', *dropout_rate_range)

        criterion = get_criterion(criterion_function)

        # Set model parameters specific fo GAT
        if model_class_name == "GAT":
            n_hidden_conv = trial.suggest_int('n_hidden_conv', 1, 3)
            hidden_conv_dim = trial.suggest_int('hidden_conv_dim', 4, 32)
            n_hidden_lin = trial.suggest_int('n_hidden_lin', 1, 3)
            hidden_lin_dim = trial.suggest_int('hidden_lin_dim', 4, 32)
            heads = trial.suggest_int('heads', 1, 3)

            gnn  = gnn_class(input_dim=input_dim, 
                            output_dim=output_dim, 
                            edge_attr_dim=edge_attr_dim,
                            n_hidden_conv=n_hidden_conv, 
                            hidden_conv_dim=hidden_conv_dim, 
                            n_hidden_lin=n_hidden_lin, 
                            hidden_lin_dim=hidden_lin_dim, 
                            dropout_rate=dropout_rate, 
                            heads=heads,
                            )
        
        # Set model parameters specific fo MessagePassing
        elif model_class_name == "MessagePassing":
            hidden_dim = trial.suggest_int('hidden_dim', 32, 256)
            num_layers = trial.suggest_int('num_layers', 2, 5)
            aggr = trial.suggest_categorical('aggr', ['mean', 'add', 'max'])

            gnn = gnn_class(input_dim=input_dim, 
                            output_dim=output_dim, 
                            edge_attr_dim=edge_attr_dim,
                            hidden_dim=hidden_dim,
                            num_layers=num_layers,
                            dropout_rate=dropout_rate,
                            aggr=aggr,
                            )
            
        # Set model parameters specific fo GraphSAGE
        elif model_class_name == "GraphSAGE":       
            n_hidden_conv = trial.suggest_int('n_hidden_conv', 1, 3)
            hidden_conv_dim = trial.suggest_int('hidden_conv_dim', 4, 32)
            n_hidden_lin = trial.suggest_int('n_hidden_lin', 1, 3)
            hidden_lin_dim = trial.suggest_int('hidden_lin_dim', 4, 32)
            jumping_knowledge = trial.suggest_categorical('jumping_knowledge', [True, False])

            gnn = gnn_class(input_dim=input_dim,
                            output_dim=output_dim,
                            edge_attr_dim=edge_attr_dim,
                            n_hidden_conv=n_hidden_conv,
                            hidden_conv_dim=hidden_conv_dim,
                            n_hidden_lin=n_hidden_lin,
                            hidden_lin_dim=hidden_lin_dim,
                            dropout=dropout_rate,
                            jumping_knowledge=jumping_knowledge,
                            )
        elif model_class_name == "GINE":    
            n_hidden_conv = trial.suggest_int('n_hidden_conv', 1, 10)   
            hidden_conv_dim = trial.suggest_int('hidden_conv_dim', 15, 40)
            n_hidden_lin = trial.suggest_int('n_hidden_lin', 1, 10)
            hidden_lin_dim = trial.suggest_int('hidden_lin_dim', 32, 50)
            jumping_knowledge = trial.suggest_categorical('jumping_knowledge', [True, False])            

            gnn = gnn_class(input_dim=input_dim,
                            output_dim=output_dim,
                            edge_attr_dim=edge_attr_dim,
                            n_hidden_conv=n_hidden_conv,
                            hidden_conv_dim=hidden_conv_dim,
                            n_hidden_lin=n_hidden_lin,
                            hidden_lin_dim=hidden_lin_dim,
                            dropout=dropout_rate,
                            jumping_knowledge=jumping_knowledge,
                            )

        print(f"GNN: \n{gnn}")

        learning_rate = trial.suggest_float('lr', *learning_rate_range)
        optimizer = optimizer_class(params=gnn.parameters(), lr=learning_rate)

        early_stop = 0
        losses = []
        val_losses = []

        for epoch in tqdm.tqdm(range(num_epochs)): #args epochs
            epoch_loss = 0.0
            epoch_val_loss = 0.0
            gnn.train()
            for batch in train_dataloader:
                epoch_loss += train_batch(data=batch, model=gnn, optimizer=optimizer, criterion=criterion)
            gnn.eval()
            for batch in val_dataloader:
                epoch_val_loss += evaluate_batch(data=batch, model=gnn, criterion=criterion)

            avg_epoch_loss = epoch_loss.item() / len(train_dataloader)
            avg_epoch_val_loss = epoch_val_loss.item() / len(val_dataloader)

            losses.append(avg_epoch_loss)
            val_losses.append(avg_epoch_val_loss)

            if epoch % 10 == 0:
                print(f'Epoch: {epoch:03d}, trn_Loss: {avg_epoch_loss:.3f}, val_loss: {avg_epoch_val_loss:.3f}')
            
            #Early stopping
            try:  
                if val_losses[-1]>=val_losses[-2]:
                    early_stop += 1
                    if early_stop == patience:
                        print("Early stopping! Epoch:", epoch)
                        break
                else:
                    early_stop = 0
            except:
                early_stop = 0
            
        
        return avg_epoch_val_loss

    # Optimize hyperparameters
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_value = study.best_value
    print(f'Best hyperparameters: {best_params}\nBest validation loss: {best_value}')


# Hyperparameter optimization heterogenous model
def hyperparams_optimization_hetero(
        train,
        val,
        n_trials=50, 
        batch_size_values=[32, 64], 
        dropout_rate_range = (0.1, 0.5),
        num_epochs=150, 
        patience=20,
        optimizer_class=torch.optim.Adam,
        criterion_function="MSELoss",
    ):
   

    def objective(trial):
        output_dims = {node_type: train[0].y_dict[node_type].shape[1] for node_type in train[0].y_dict.keys()}
        batch_size = trial.suggest_categorical('batch_size', batch_size_values)
        train_dataloader = pyg_DataLoader(train, batch_size=batch_size, shuffle=True)
        val_dataloader = pyg_DataLoader(val, batch_size=batch_size, shuffle=False)

        gnn_class = get_gnn('HeteroGNN')

        dropout_rate = trial.suggest_float('dropout_rate', *dropout_rate_range)
        criterion = get_criterion(criterion_function)
        n_hidden_conv = trial.suggest_int('n_hidden_conv', 1, 5)   
        hidden_conv_dim = trial.suggest_int('hidden_conv_dim', 64, 1024)
        heads = trial.suggest_int('heads', 1, 4)
        n_hidden_lin = trial.suggest_int('n_hidden_lin', 1, 5)
        hidden_lin_dim = trial.suggest_int('hidden_lin_dim', 64, 1024)
        jumping_knowledge = trial.suggest_categorical('jumping_knowledge', [True, False]) 
        hetero_aggr = trial.suggest_categorical('hetero_aggr', ['mean', 'max', 'min', 'sum'])           

        gnn = gnn_class(output_dim_dict=output_dims,
                        edge_types=train[0].edge_index_dict.keys(),
                        n_hidden_conv=n_hidden_conv,
                        hidden_conv_dim=hidden_conv_dim,
                        n_heads=heads,
                        n_hidden_lin=n_hidden_lin,
                        hidden_lin_dim=hidden_lin_dim,
                        dropout_rate=dropout_rate,
                        jumping_knowledge=jumping_knowledge,
                        hetero_aggr=hetero_aggr,
                        )

        print(f"GNN: \n{gnn}")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Current device: {device}")
        gnn = gnn.to(device)

        optimizer = optimizer_class(params=gnn.parameters(), lr=1e-4)

        early_stop = 0
        losses = []
        val_losses = []

        for epoch in tqdm.tqdm(range(num_epochs)): #args epochs
            epoch_loss = 0.0
            epoch_val_loss = 0.0
            gnn.train()
            for batch in train_dataloader:
                epoch_loss += train_batch_hetero(data=batch, model=gnn, optimizer=optimizer, criterion=criterion, device=device)
            gnn.eval()
            for batch in val_dataloader:
                epoch_val_loss += evaluate_batch_hetero(data=batch, model=gnn, criterion=criterion, device=device)

            avg_epoch_loss = epoch_loss.item() / len(train_dataloader)
            avg_epoch_val_loss = epoch_val_loss.item() / len(val_dataloader)

            losses.append(avg_epoch_loss)
            val_losses.append(avg_epoch_val_loss)

            if epoch % 10 == 0:
                print(f'Epoch: {epoch:03d}, trn_Loss: {avg_epoch_loss:.3f}, val_loss: {avg_epoch_val_loss:.3f}')
            
            #Early stopping
            try:  
                if val_losses[-1]>=val_losses[-2]:
                    early_stop += 1
                    if early_stop == patience:
                        print("Early stopping! Epoch:", epoch)
                        break
                else:
                    early_stop = 0
            except:
                early_stop = 0
            
        
        return avg_epoch_val_loss

    # Optimize hyperparameters
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_value = study.best_value
    print(f'Best hyperparameters: {best_params}\nBest validation loss: {best_value}')


if __name__ == "__main__":
    print("Loading Data")
    # Make sure to change it to the correct path on your data
    train, val, _ = load_data("./Data/train","./Data/val","./Data/test", "HeteroGNN")
    print(f"Data Loaded \n",
          f"Number of training samples = {len(train)}\n",
          f"Number of validation samples = {len(val)}\n")

    hyperparams_optimization_hetero(train=train, val=val)