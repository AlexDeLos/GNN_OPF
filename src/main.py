from train import train_model
from utils import get_arguments, load_data, save_model, plot_losses
import warnings

# Suppress FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    print("Parsing Arguments")
    arguments = get_arguments()
    print(f"Parsed arguments: {arguments}")

    
    train, val, test = load_data(arguments.train, arguments.val, arguments.test)    

    print(f"Data Loaded \n",
          f"Number of training samples = {len(train)}\n",
          f"Number of validation samples = {len(val)}\n",
          f"Number of testing samples = {len(test)}\n",)

    print("Training Model")
    model, losses, val_losses = train_model(arguments, train, val, test)
    model_class_name = model.class_name
    if arguments.save_model:
        print("Saving Model")
        save_model(model, arguments.model_name, model_class_name)
    if arguments.plot:
        plot_losses(losses, val_losses, model_class_name)
    
    
if __name__ == "__main__":
    main()