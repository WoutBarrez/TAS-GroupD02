import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from values import *

def read_mic_positions():
    mic_positions_file = 'gps/mic_positions.csv' #reading file for the gps
    mic_positions = pd.read_csv(mic_positions_file) #read the values using pandas
    x_positions = mic_positions['x'].tolist()
    y_positions = mic_positions['y'].tolist()

    return x_positions,y_positions #returns name index and the pandas read of gps values

def plot_coordinates(x_positions, y_positions):
    plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
    plt.scatter(x_positions, y_positions, color='blue', label='Positions')
    plt.title('Plot of Coordinates')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.legend()
    plt.show()

def rotate_positions(x_positions, y_positions, drone):
    rotated_x_positions=[]
    rotated_y_positions=[]
    #angle=np.pi/180*1
    angle=(array_angle[drone]+90)*np.pi/180
    for i in range(len(x_positions)):
        rotated_x_positions.append(x_positions[i]*np.cos(angle)-y_positions[i]*np.sin(angle))
        rotated_y_positions.append(x_positions[i]*np.sin(angle)+y_positions[i]*np.cos(angle))
    
    return rotated_x_positions, rotated_y_positions


x_positions, y_positions= read_mic_positions()

plot_coordinates(x_positions, y_positions)

new_x_positions, new_y_positions = rotate_positions(x_positions, y_positions, "Phantom")

def plot_coordinates_comparison(x_positions, y_positions, new_x_positions, new_y_positions):
    plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
    plt.scatter(x_positions, y_positions, color='blue', label='Positions')
    plt.scatter(new_x_positions, new_y_positions, color='red', label='Corrected positions')
    plt.title('Plot of Coordinates')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.legend()
    plt.show()

plot_coordinates_comparison(x_positions, y_positions, new_x_positions, new_y_positions)