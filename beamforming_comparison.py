import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
import os


drone="Autel_Evo"


#Grid point characteristics

class Grid_point:
    def __init__(self, r, theta, phi, beam, g):
        self.r = r
        self.theta = theta
        self.phi = phi
        self.beam = beam
        self.g = g

class Microphone:
    def __init__(self, r, theta, phi):
        self.r = r
        self.theta = theta
        self.phi = phi

#Calculate the steering vectors 

def get_steering_vector(scan_r, scan_theta, scan_phi, F, C):
    g = []
    for i in range(len(m)):
        dist = np.sqrt((scan_r*np.sin(scan_phi*np.pi/180))**2 +
                    (scan_r*np.cos(scan_phi*np.pi/180)*np.cos(scan_theta*np.pi/180) - m[i].r*np.cos(m[i].theta*np.pi/180))**2 +
                    (scan_r*np.cos(scan_phi*np.pi/180)*np.sin(scan_theta*np.pi/180) - m[i].r*np.sin(m[i].theta*np.pi/180))**2)
        z = np.exp(-(2*np.pi*F*dist*complex(0+1j)/C))/dist
        g.append(z)
    g = np.array(g)
    return g

#Main program

#Read pressure measurements

p = [] 

with open(f"C:\\Users\\Jakub\\OneDrive - Delft University of Technology\\Desktop\\School\\test analysis project\\Microphone data\\{drone}.csv", newline='') as drone_file:
    reader = csv.reader(drone_file)
    for row in reader:
        if len(row[0])==0:
            break
        aux = []
        for i in range(len(row)):
            if i!=0 and i!=16 and i!=20 and i!=40 and i<62:
                aux.append(float(row[i]))
        p.append(aux)

p = np.array(p) #Columns (p) are p measurements for the same microphone and diff time instances
          #and rows (p[i]) are for same time instance and different microphones

#Read microphone coordinates

m = [] #Microphone coordinates (r, theta, phi) wrt the center of the array

with open("config.txt") as m_file:
    for line in m_file:
        words = line.split()
        if words[0]!="1" and words[0]!="17" and words[0]!="21" and words[0]!="41" and words[0]!="63" and words[0]!="64":
            x = float(words[1])
            y = float(words[2])
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y, x)
            theta = theta if theta>=0 else theta + 2*np.pi
            microphone = Microphone(r, theta*180/np.pi, 0)
            m.append(microphone)

#Construct the grid
grid = []

dtheta = 1
dphi = 0.5
phi = 0
r = 100

while phi < 90:
    theta = 0
    while theta < 360:
        grid_point = Grid_point(r, theta, phi, 0, [])
        grid.append(grid_point)
        theta += dtheta
    phi += dphi

#Prepare the plot
    
x_axis = [] #Horizontal axis is theta
y_axis = [] #Vertical axis is phi

theta = 0
while theta < 360:
    x_axis.append(theta)
    theta += dtheta

x_axis.append(360)

phi = 0
while phi < 90:
    y_axis.append(phi)
    phi += dphi
y_axis.append(90)

X,Y = np.meshgrid(x_axis, y_axis)

#Construct the steering vectors

natural_frequency=180

class Result:
    def __init__(self, N, F, e):
        self.N = N
        self.F = F
        self.e = e

results=[]

file_path = f'beamforming_values/{drone}_errors.csv'

# Check if the file exists
if not os.path.exists(file_path):
    # Define the column names
    column_names = ['N', 'F', 'e']

    # Create an empty DataFrame with the specified column names
    df = pd.DataFrame(columns=column_names)

    # Save the DataFrame to a CSV file
    df.to_csv(file_path, index=False)

    print(f"CSV file for {drone} created successfully!")
else:
    print(f"File for {drone} already exists.")

existing_file=pd.read_csv(f'beamforming_values/{drone}_errors.csv')

N_list=existing_file['N'].tolist()
F_list=existing_file['F'].tolist()
e_list=existing_file['e'].tolist()

#merging N and F lists
merge_list=[]
for i in range(len(N_list)):
    merge_list.append([N_list[i], F_list[i]])

for N in range(1,11):
    for F in range(natural_frequency, 10*natural_frequency, natural_frequency):
        is_already_there=False
        print(f"N={N}")
        print(f"F={F}")

        for i in range(len(merge_list)):
            if [N, F] == merge_list[i]:
                is_already_there=True
        if is_already_there==False:

            N_list=existing_file['N'].tolist()
            F_list=existing_file['F'].tolist()
            e_list=existing_file['e'].tolist()

            #Define constants
            C = 343 #m/s - speed of sound
            #F = 690 #Hz - frequency needed for the steering vectors and fft
            #N = 7 #frequency bands

            fs = 50000
            wind = 4096 #window size
            df = (fs/2)/(wind/2)

            n = round(F/df)

            #Calculate the steering vectors for each grid point

            #for i in range(len(grid)):
            #    grid[i].g = get_steering_vector(grid[i].r, grid[i].theta, grid[i].phi, n*df, C)

            for i in range(len(grid)):
                ster_vector = []
                for j in range(N):
                    ster_vector.append(get_steering_vector(grid[i].r, grid[i].theta, grid[i].phi, (n+N//2-j)*df, C))
                
                grid[i].g = ster_vector

            #Apply the beamforming fomrula
            theta_max = []
            phi_max  = []
            time = []

            for i in range(0, len(p) - wind, wind//2):
                
                for j in range(len(grid)):
                    grid[j].beam = 0

                for k in range(N):
                    p_fft = np.fft.fft(p[i:i+wind], axis=0)[n+1+N//2-k] 
                    p_star = np.conj(p_fft) 

                    aux = np.outer(p_fft, p_star)

                    for j in range(len(grid)):
                        g_star = np.conj(grid[j].g[k])
                        g_norm_squared = (np.linalg.norm(grid[j].g[k]))**2
                        aux2 = (aux.T @ g_star)
                        aux3 = aux2 @ grid[j].g[k]
                        grid[j].beam += (aux3/g_norm_squared) / N
                
                beam_map = np.zeros((len(y_axis)-1, len(x_axis)-1))
                index = 0

                for j in range(len(y_axis)-1):
                    for k in range(len(x_axis)-1):
                        beam_map[j][k] = grid[index].beam
                        index += 1
                
                #Printing
                max_value = 0
                x_max = 0
                y_max = 0
                for j in range(len(grid)):
                    if grid[j].beam > max_value:
                        x_max = grid[j].theta
                        y_max = grid[j].phi
                        max_value = grid[j].beam
                    
                #print(f"At time instant {i/50000}: o = {x_max}, f = {y_max}.")
                theta_max.append(x_max)
                phi_max.append(y_max)
                time.append(i/50000)    

                #Plotting    
                    
                #fig, ax = plt.subplots()
                #ax.pcolormesh(X, Y, beam_map)
                #fig.canvas.manager.set_window_title(f"Time instant {0.00002*i}")
                #print(beam_map)
                #plt.show()

            #Plot the trajectory of the sound source

            e=0

            for i in range(len(theta_max)-1):
                if np.abs(theta_max[i+1]-theta_max[i])>2:
                    e=e+1

            for i in range(len(phi_max)-1):
                if np.abs(phi_max[i+1]-phi_max[i])>2:
                    e=e+1

            results.append(Result(N, F, e))

            plt.scatter(time, theta_max, label="Theta")
            plt.xlabel("Time [s]")
            plt.ylabel("Theta [deg]")
            plt.savefig(f'C:\\Users\\Jakub\\OneDrive - Delft University of Technology\\Desktop\\School\\test analysis project\\{drone} comparison\\{drone} theta output F={F} N={N}')
            #plt.show()
            plt.clf()

            plt.scatter(time, phi_max, label="Phi")
            plt.xlabel("Time [s]")
            plt.ylabel("Phi [deg]")
            plt.savefig(f'C:\\Users\\Jakub\\OneDrive - Delft University of Technology\\Desktop\\School\\test analysis project\\{drone} comparison\\{drone} phi output F={F} N={N}')
            #plt.show()
            plt.clf()

            existing_file=pd.read_csv(f'beamforming_values/{drone}_errors.csv')

            N_list=existing_file['N'].tolist()
            F_list=existing_file['F'].tolist()
            e_list=existing_file['e'].tolist()

            N_list.append(N)
            F_list.append(F)
            e_list.append(e)

            e_init=99999
            for i in range(len(results)):
                if results[i].e<e_init:
                    best=results[i]
                    e_init=results[i].e

            df = pd.DataFrame({
                'N': N_list,
                'F': F_list,
                'e': e_list
            })

            df.to_csv(f'beamforming_values/{drone}_errors.csv', index=False)

            print(f"Saved for N={N} F={F}")

        else:
            print(f"F={F}, N={N} is already saved, skipping...")

print(f"N={best.N}, F={best.F}, errors={best.e}")
