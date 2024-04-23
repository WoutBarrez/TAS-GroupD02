from functions_drone_gps import *

drone="ANWB"

gps_read=read(drone)

gps_read=avoid_errors(gps_read)

#haversine

x_haversine, y_haversine, altitudes, time= give_and_convert_coordinates_haversine(gps_read, drone)

x_haversine, y_haversine, altitudes, time= take_sample(x_haversine, y_haversine, altitudes, time, drone)

#old

x_mercator, y_mercator, altitudes, time= give_and_convert_coordinates_mercator(gps_read, drone)

x_mercator, y_mercator, altitudes, time= take_sample(x_mercator, y_mercator, altitudes, time, drone)


#rotated_x_positions, rotated_y_positions = rotate_positions(x_positions, y_positions, drone)

plot_coordinates_comparison(x_mercator, y_mercator, x_haversine, y_haversine)

#theta, phi, r = convert_to_polar(rotated_x_positions, rotated_y_positions, altitudes)

#print(r)

#save_data_to_csv(time, theta, phi, drone)