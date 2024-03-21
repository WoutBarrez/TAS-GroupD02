from functions_drone_gps import *

drone="Altura"

gps_read=read(drone)

gps_read=avoid_errors(gps_read)

x_positions, y_positions, altitudes= give_and_convert_coordinates(gps_read, drone)

rotated_x_positions, rotated_y_positions = rotate_positions(x_positions, y_positions, drone)

#plot_coordinates_comparison(x_positions, y_positions, rotated_x_positions, rotated_y_positions)

theta, phi = convert_to_polar(rotated_x_positions, rotated_y_positions, altitudes)

plot_coordinates(theta, phi)