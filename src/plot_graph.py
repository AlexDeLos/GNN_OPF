import os
import sys
# local imports
# add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils_plot import distance_plot, plot_losses, plot_percent_curve

def main():
    csv_dict = {
        # 'physics_loss_-0' : 'Data/results/physics_0_results(deep).csv',
        'physics_loss_-1' : 'Data/results/physics_1_results(deep).csv',
        # 'physics_loss_-2' : 'Data/results/physics_2_results(deep).csv',
        'physics_loss_-3' : 'Data/results/physics_3_results(deep).csv',
        'physics_loss_-10' : 'Data/results/physics_10_results(deep).csv',
        # 'physics_loss_-15' : 'Data/results/physics_15_results(new_column).csv',
        # 'physics_loss_-20' : 'Data/results/physics_20_results(new_column).csv',
    }

    # load_vm_pu
    # load_va_deg
    # gen_va_deg
    # load_gen_va_deg
    plot_percent_curve(csv_dict, col_name= 'load_vm_pu', colors=['red', 'blue', 'green', 'orange', 'purple', 'black', 'grey'])
    plot_percent_curve(csv_dict, col_name= 'va_degree', colors=['red', 'blue', 'green', 'orange', 'purple', 'black', 'grey'])
    # plot_percent_curve(csv_dict, col_name= 'load_va_deg', colors=['red', 'blue', 'green', 'orange', 'purple', 'black', 'grey'])
    # plot_percent_curve(csv_dict, col_name= 'gen_va_deg', colors=['red', 'blue', 'green', 'orange', 'purple', 'black', 'grey'])
    # plot_percent_curve(csv_dict, col_name= 'load_gen_va_deg', colors=['red', 'blue', 'green', 'orange', 'purple', 'black', 'grey'])

main()