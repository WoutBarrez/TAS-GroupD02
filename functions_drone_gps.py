import pandas as pd
import numpy as np
from values import *
import matplotlib.pyplot as plt

def read(drone_name):
    values_gps = 'gps/GPS_consistent/'+drone_name+'_GPS.csv' #reading file for the gps
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

    return df_filtered

def convert_coordinates(drone_name):

    gps_read=read(drone_name)

    latitudes = gps_read['Latitude'].tolist()
    longitudes = gps_read['Longitude'].tolist()

    lat0=array_location[drone_name][0]
    lon0=array_location[drone_name][1]

    x=[]
    y=[]

    for i in range(len(latitudes)):

        lon=longitudes[i]
        lat=latitudes[i]

        x.append((lon-lon0)*40000*np.cos((lat+lat0)*np.pi/360)/360)
        y.append((lat-lat0)*40000/360)

    return x,y

def rotate_positions(x_positions, y_positions, drone):
    rotated_x_positions=[]
    rotated_y_positions=[]
    #angle=np.pi/180*1
    angle=-array_angle[drone]*np.pi/180
    for i in range(len(x_positions)):
        rotated_x_positions.append(x_positions[i]*np.cos(angle)-y_positions[i]*np.sin(angle))
        rotated_y_positions.append(x_positions[i]*np.sin(angle)+y_positions[i]*np.cos(angle))
    
    return rotated_x_positions, rotated_y_positions

def plot_coordinates(x_positions, y_positions):
    plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
    plt.scatter(x_positions, y_positions, color='blue', label='Positions')
    plt.title('Plot of Coordinates')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.legend()
    plt.show()

x_positions, y_positions= convert_coordinates("ANWB")
rotated_x_positions, rotated_y_positions = rotate_positions(x_positions, y_positions, drone)

plot_coordinates(x_positions, y_positions)