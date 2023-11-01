import os
import sys
# local imports
# add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils_plot import distance_plot, plot_losses, plot_percent_curve

def main():
    csv_dict = {
        'physics_loss_-0' : 'Data/results/physics_0_results.csv',
        'physics_loss_-1' : 'Data/results/-1(test).csv',
        'physics_loss_-5' : 'Data/results/physics_5_results.csv',
    }

    plot_percent_curve(csv_dict, col_name= 'load_vm_pu')

main()