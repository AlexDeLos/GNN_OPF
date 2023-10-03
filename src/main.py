import argparse
import os
import pandas as pd
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import random
import string

import networkx as nx
import pandapower as pp
import pandapower.plotting as ppl
from architectures.GAT import GATNodeRegression

from models.GATConv import GATConvolution
# from playground.Loss_Playground import Loss_Playground

import torch as th
import torch.nn as nn
from torch_geometric.utils.convert import from_networkx
from torch_geometric.loader import DataLoader as pyg_DataLoader
from train import train_model
from utils import get_arguments, load_data, save_model, plot_losses
import warnings

# Suppress FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    print("Parsing Arguments")
    arguments = get_arguments()
    print(f"Parsed arguments: {arguments}")

    print("Loading Training Data")
    train = load_data(arguments.train)
    print("Loading Validation Data")
    val = load_data(arguments.val)
    print("Loading Testing Data")
    test = load_data(arguments.test)

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



