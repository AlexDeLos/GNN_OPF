from train import train_model
from utils import get_arguments, load_data, normalize_data, save_model
from plot_utils import distance_plot, plot_losses
import warnings

# Suppress FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    print("Parsing Arguments")
    arguments = get_arguments()
    print(f"Parsed arguments: {arguments}")

    
    train, val, test = load_data(arguments.train, arguments.val, arguments.test, load_physics=arguments.physics)

    if arguments.normalize:
        print("Normalizing Data")
        train, val, test = normalize_data(train, val, test)

    print(f"Data Loaded \n",
          f"Number of training samples = {len(train)}\n",
          f"Number of validation samples = {len(val)}\n")
        #   f"Number of testing samples = {len(test)}\n",)

    print("Training Model")
    
    model, losses, val_losses, last_batch = train_model(arguments, train, val)

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



