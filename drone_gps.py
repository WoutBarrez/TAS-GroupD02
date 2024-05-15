from functions_drone_gps import *

drone="Autel_Evo"

gps_read=read(drone)

gps_read=avoid_errors(gps_read)

#haversine

theta_h, phi_h, r_h, time_h= give_and_convert_coordinates_haversine(gps_read, drone)

#old

x_mercator, y_mercator, x_mercator2, y_mercator2, altitudes_m, time_m= give_and_convert_coordinates_mercator(gps_read, drone)

x_mercator, y_mercator, altitudes, time= take_sample(x_mercator, y_mercator, altitudes_m, time_m, drone)

rotated_x_positions, rotated_y_positions = rotate_positions(x_mercator, y_mercator, drone)

#rotated_x_positions2, rotated_y_positions2 = rotate_positions(x_mercator2, y_mercator2, drone)

theta, phi, r = convert_to_polar(rotated_x_positions, rotated_y_positions, altitudes_m)

#theta2, phi2, r2 = convert_to_polar(rotated_x_positions2, rotated_y_positions2, altitudes_m)

theta_func=create_interpolation(time_m, theta)
phi_func=create_interpolation(time_m, phi)

values = 'beamforming_values/'+drone+'.csv' #reading file for the gps
data = pd.read_csv(values) #read the values using pandas
thetas = data['Theta'].tolist()
phis = data['Phi'].tolist()
times = data['Time'].tolist()

thetas=[-x+60 for x in thetas]
thetas=[x - 360 if x > 180 else x for x in thetas]
#thetas=[-x for x in thetas]

#phis_plus_5 = [phi + 6 for phi in phis]


#rotated_x_positions, rotated_y_positions = rotate_positions(x_positions, y_positions, drone)

#plot_coordinates_with_endpoints(rotated_x_positions2, rotated_y_positions2)

#plot_coordinates_comparison(rotated_x_positions, rotated_y_positions, rotated_y_positions, rotated_y_positions2)

plot_angles_comparison(time_m, theta, times, thetas)

#theta, phi, r = convert_to_polar(rotated_x_positions, rotated_y_positions, altitudes)

#print(r)

#save_data_to_csv(time, theta, phi, drone)