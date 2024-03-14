import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from values import *

def read(drone_name):
    values_gps = 'gps/GPS/'+drone_name+'_GPS.csv' #reading file for the gps
    gps_read = pd.read_csv(values_gps) #read the values using pandas
    return gps_read #returns name index and the pandas read of gps values

def avoid_errors(gps_read):
    df = pd.DataFrame(gps_read)

    # Specify the specific values you want to exclude
    values_to_exclude = [0, None]

    # Create a boolean mask for rows containing specific values or NaN
    mask = ~(df.isin(values_to_exclude) | df.isna()).any(axis=1)

    # Use the mask to filter the DataFrame and keep rows without specific values
    df_filtered = df[mask]

    print(df_filtered)

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
    new_x_positions=[]
    new_y_positions=[]
    #angle=-np.pi/180*1
    angle=-array_angle[drone]*np.pi/180
    for i in range(len(x_positions)):
        new_x_positions.append(x_positions[i]*np.cos(angle)-y_positions[i]*np.sin(angle))
        new_y_positions.append(x_positions[i]*np.sin(angle)+y_positions[i]*np.cos(angle))
    
    return new_x_positions, new_y_positions


x_positions, y_positions= read_mic_positions()

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