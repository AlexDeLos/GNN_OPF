from train import train_model
from train_hetero import train_model_hetero
from utils import get_arguments, load_data, normalize_data, save_model
from utils_hetero import normalize_data_hetero
from plot_utils import distance_plot, plot_losses
import warnings

# Suppress FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    print("Parsing Arguments")
    arguments = get_arguments()
    print(f"Parsed arguments: {arguments}")

    
    train, val, test = load_data(arguments.train, arguments.val, arguments.test, arguments.gnn)  

    if arguments.normalize:
        print("Normalizing Data")
        if arguments.gnn[:6] != "Hetero":
            train, val, test = normalize_data(train, val, test)
        else: 
            train, val, test = normalize_data_hetero(train, val, test)

    print(f"Data Loaded \n",
          f"Number of training samples = {len(train)}\n",
          f"Number of validation samples = {len(val)}\n")
        #   f"Number of testing samples = {len(test)}\n",)

    print("Training Model")

    if arguments.gnn[:6] != "Hetero":
        model, losses, val_losses, last_batch = train_model(arguments, train, val)
    else:
        model, losses, val_losses, last_batch = train_model_hetero(arguments, train, val)

    if arguments.save_model:
        print("Saving Model")
        save_model(model, arguments.model_name)
    if arguments.plot:
        print("Plotting losses")
        plot_losses(losses, val_losses, model.class_name)
    
    if arguments.plot_node_error:
        print("Plotting node error per distance from generator")
        distance_plot(model, last_batch)
    
    
if __name__ == "__main__":
    main()