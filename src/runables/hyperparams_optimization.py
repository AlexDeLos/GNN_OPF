import optuna
import torch
from torch_geometric.loader import DataLoader as pyg_DataLoader
import tqdm
from train_homo import train_batch, evaluate_batch
from utils import get_gnn, load_data, get_criterion
from utils_homo import normalize_data

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


if __name__ == "__main__":
    print("Loading Data")
    # Make sure to change it to the correct path on your data
    train, val, test = load_data("./Data_sanity3(rnd_walk)/train","./Data_sanity3(rnd_walk)/val","./Data_sanity3(rnd_walk)/val")
    train, val, test = normalize_data(train, val, test)
    print(f"Data Loaded \n",
          f"Number of training samples = {len(train)}\n",
          f"Number of validation samples = {len(val)}\n",
          f"Number of testing samples = {len(test)}\n",)
    
    hyperparams_optimization(train=train, model_class_name="GINE")