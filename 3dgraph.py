import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

drone="Autel_Evo"

def plot_3d_graph(x_data, y_data, z_data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_data, y_data, z_data, c='r', marker='o')  # Scatter plot
    ax.set_xlabel('N')
    ax.set_ylabel('F')
    ax.set_zlabel('Error')
    plt.show()

existing_file=pd.read_csv(f'beamforming_values/{drone}_errors.csv')

# Example usage:
N_list=existing_file['N'].tolist()
F_list=existing_file['F'].tolist()
e_list=existing_file['e'].tolist()

F_list=[F/180 for F in F_list]

plot_3d_graph(N_list, F_list, e_list)