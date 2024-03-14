import numpy as np
import csv
import matplotlib.pyplot as plt
#import time

#Transform microphone positions to spherical coordinates

def get_spherical_from_cart(m):
    n = []
    for i in range(len(m)):
        r = np.sqrt(m[i][0]**2 + m[i][1]**2)
        theta = np.arctan2(m[i][1], m[i][0])
        n.append([r,theta*180/np.pi,90])
    return n

#Create hemisphere

def get_hemisphere():
    dtheta = 1
    dphi = 0.5
    phi = 90
    r = 1
    grid = []
    while phi > 0:
        theta = -180
        while theta < 180:
            grid.append([r,theta, phi])
            theta += dtheta
        phi -= dphi
    #grid.append([r,0,0])
    return grid

#Calculate the steering vectors 

def get_steering_vector(scan_r, scan_theta, scan_phi):
    g = []
    for i in range(len(m)):
        r = np.sqrt((scan_r)**2 + (m[i][0])**2 - 
                    2*scan_r*m[i][0]*(np.sin(scan_theta*np.pi/180)*np.sin(m[i][1]*np.pi/180)*np.cos((scan_phi - m[i][2])*np.pi/180) 
                                      + np.cos(scan_theta*np.pi/180)*np.cos(m[i][1]*np.pi/180)))
        z = complex(np.cos(2*np.pi*F*r/C), -np.sin(2*np.pi*F*r/C))
        g.append(z/r)
    g = np.array(g)
    return g

#Get conjugate transpose of a vector

def get_conjugate(x):
    y = []
    for i in range(len(x)):
        y.append(np.conj(x[i]))
    return y

#Main program

#Define constants
C = 343 #m/s - speed of sound
F = 300 #Hz - frequency obtained from spectogram
R = 1 #m - the radius of the hemisphere

#Read pressure measurements

p = [] 

with open("D:\carti2022\Python\ANWB.csv", newline='') as drone_file:
    reader = csv.reader(drone_file)
    for row in reader:
        if len(row[0])==0:
            break
        aux = []
        for i in range(len(row)):
            if i!=0 and i!=16 and i!=20 and i!=40 and i<62:
                aux.append(float(row[i]))
        p.append(aux)

p = np.array(p) #Columns are p measurements for the same microphones and diff time instances
          #and rows are for same time instance and different microphones

#Read microphone coordinates

m = [] #Microphone coordinates (x,y) wrt the center of the array

with open("D:\carti2022\Python\config.txt") as m_file:
    for line in m_file:
        words = line.split()
        if words[0]!="1" and words[0]!="17" and words[0]!="21" and words[0]!="41" and words[0]!="63" and words[0]!="64":
            m.append([float(words[1]),float(words[2])])

#Prepare for beamforming
m = get_spherical_from_cart(m)
grid = get_hemisphere()

#Apply the beamforming formula

x_axis = set()
y_axis = set()
for j in range(len(grid)):
    x_axis.add(grid[j][1])
    y_axis.add(grid[j][2])

x_axis.add(180)
y_axis.add(0)
x_axis = sorted(x_axis)
y_axis = sorted(y_axis)
X,Y = np.meshgrid(x_axis, y_axis)

g = []
g_star = []
g_norm_squared = []

for i in range(len(grid)):
    g.append(get_steering_vector(grid[i][0], grid[i][1], grid[i][2]))
    g_star.append(np.conjugate(g[-1]))
    g_norm_squared.append((np.linalg.norm(g[-1]))**2)

for i in range(0, len(p)-4096, 2048):
    #start = time.time()
    p_fft = np.fft.fft(p[i:i+4096].T)
    p_star = np.conj(p_fft)
    #end1 = time.time()
    #for k in range(len(p_fft)):
    #aux = np.outer(p_fft,p_star)
    aux = p_fft @ p_star.T
    b = []
    for j in range(len(g)):
            #g = get_steering_vector(grid[j][0], grid[j][1], grid[j][2])
            #g_star = get_conjugate(g)
        aux2 = (aux.T @ g_star[j]).T
        aux3 = aux2 @ g[j]
            #g_norm_squared = (np.linalg.norm(g))**2
        b.append((aux3/g_norm_squared[j]).real)
    b = np.reshape(b,(len(y_axis)-1,len(x_axis)-1))
    fig, ax = plt.subplots()
    ax.pcolormesh(x_axis, y_axis, b)
    fig.canvas.manager.set_window_title(f"Time instant {0.00002*i}")
    plt.show()
        #end2 = time.time()
        #print(end1 - start, end2 - end1)
        #plt.show(block=False)
        #plt.pause(1)
        #plt.close()
