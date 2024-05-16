## ----------------------------------------------------------------------------------------------
## This file only contains the main code to classify the data and to execute the classifier. The
## data files are separate. The code pertaining to the actual classifier are located in the
## SVM_classifier file. Both the data files and the SVM_classifier file are required in order to
## run this program.
## ----------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------
# Import necessary libraries
# -----------------------------------------------------------------------------------------------
import csv
import numpy as np
import SVM_classifier
from sklearn.model_selection import train_test_split
import os
import torch

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

# -------------------------------------------------------------------------------------------
# Defining a function to slice the data into a specified number (percentage - sample_percent)
# of separate datasets e.g. 4 training datasets and 1 test data set, or 3 training data sets 
# and 2 test data sets.
# This function is used in training the AI model and in feeding the model the test data.
# -------------------------------------------------------------------------------------------
sample_percent = 0.01
def data_clean(drone, sample_percent):
    X_data = []
    data_arr = np.array(drone).T
    total_time_points_no = len(data_arr)
    sample_size = int(total_time_points_no * sample_percent)
    for i in range(0,int(1/sample_percent)):
        X = data_arr[sample_size*i:sample_size*(i+1)].T
        X_data  += [X]
    return np.array(X_data)

# -----------------------------------------------------------------------------------------------
# Defining a function to train the classifier, this will be called later on if the classifier
# has yet to be trained.
# -----------------------------------------------------------------------------------------------
def AI():
    # -------------------------------------------------------------------------------------------
    # Sample the data representing air pressure readings for each drone.
    # Each row represents a drone, and each column represents a feature (e.g., microphone array
    # readings).
    # -------------------------------------------------------------------------------------------
    data_Altura = []
    data_ANWB = []
    data_ATMOS = []
    data_Autel_Evo = []
    data_Phantom = []
    name_dic = {'Altura.csv': data_Altura, 'ANWB.csv': data_ANWB, 'ATMOS.csv': data_ATMOS, 'Autel_Evo.csv': data_Autel_Evo, 'Phantom.csv': data_Phantom}

    def datasets():
        data_lst = ['Altura.csv', 'ANWB.csv', 'ATMOS.csv', 'Autel_Evo.csv', 'Phantom.csv']
        for filename in data_lst:
            print(f'Reading {filename} database...')
            data = []
            with open(filename, "r") as file:
                csvreader = csv.reader(file)
                for row in csvreader:
                    row1 = []
                    for i,e in enumerate(row):
                        row1.append(float(e))
                    data.append(row1)
            step = np.array(data[:249999])
            for i,row in enumerate(step.T):
                if i != 0 and i != 16 and  i !=20 and i != 40 and  i !=62 and  i !=63:
                    name_dic[filename].append(row)

    datasets()

    X_data_Altura = data_clean(name_dic['Altura.csv'], sample_percent)
    X_data_ANWB = data_clean(name_dic['ANWB.csv'], sample_percent)
    X_data_ATMOS = data_clean(name_dic['ATMOS.csv'], sample_percent)
    X_data_Autel_Evo = data_clean(name_dic['Autel_Evo.csv'], sample_percent)
    X_data_Phantom = data_clean(name_dic['Phantom.csv'], sample_percent)

    # -------------------------------------------------------------------------------------------
    # Split the data into training and testing sets
    # -------------------------------------------------------------------------------------------
    X_data = []
    y_data = []
    for i in X_data_Altura:
        for e in i:
            X_data += [e]
            y_data += [0]
    for i in X_data_ANWB:
        for e in i:
            X_data += [e]
            y_data += [1]
    for i in X_data_ATMOS:
        for e in i:
            X_data += [e]
            y_data += [2]
    for i in X_data_Autel_Evo:
        for e in i:
            X_data += [e]
            y_data += [3]
    for i in X_data_Phantom:
        for e in i:
            X_data += [e]
            y_data += [4]

    #X_data = torch.from_numpy(np.array(X_data)).float()
    #y_data = torch.from_numpy(np.array(y_data)).float()
    train, test, train_labels, test_labels = train_test_split(X_data, y_data, test_size=0.20, random_state=42)

    # Create an SVM classifier
    n_features = len(X_data[1])
    n_hidden_neurons = 50 #120 
    n_classes = 5
    learning_rate = 0.02 #0.008
    n_epochs = 1000 #3000
    parameters = [n_features, n_hidden_neurons, n_classes, learning_rate, n_epochs]
    with open("Model_parameters.txt", "w") as file:
        for i in range(5):
            # Writing data to a file
            file.write(str(parameters[i]) + '\n')
    clf = SVM_classifier.MLPClassifier(n_features, n_hidden_neurons, n_classes, learning_rate, n_epochs)

    # -------------------------------------------------------------------------------------------
    # Train the classifier
    # -------------------------------------------------------------------------------------------
    clf.train(train, train_labels)
    torch.save(clf.state_dict(), 'SVM_model')

    return clf, test_labels

# -----------------------------------------------------------------------------------------------
# Read the provided data file containing the unknown drone
# -----------------------------------------------------------------------------------------------
def readfile():
    data_drone = []
    data = []
    with open(dir_path + '\Test data\Test file.csv', "r") as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            row1 = []
            for i,e in enumerate(row):
                row1.append(float(e))
            data.append(row1)
        step = np.array(data[:249999])
        for i,row in enumerate(step.T):
            if i != 0 and i != 16 and  i !=20 and i != 40 and  i !=62 and  i !=63:
                data_drone.append(row)
    X_data = []
    for i in data_clean(data_drone, sample_percent):
        for e in i:
            X_data += [e]
    return X_data

run = os.path.isfile(dir_path + '\SVM_model')
#print(dir_path + 'Test data') 

# -----------------------------------------------------------------------------------------------
# Running the classifier
# -----------------------------------------------------------------------------------------------
if run == False:
    print('The AI must first be trained. Reading files:')
    model, test_labels = AI()
    print(test_labels)
elif run == True:
    print('The model has already been trained. Proceeding with identification of drone...')
    n_features = len(readfile()[1])
    n_hidden_neurons = 50 #120 
    n_classes = 5
    learning_rate = 0.02 #0.008
    n_epochs = 1000 #3000
    model = SVM_classifier.MLPClassifier(n_features, n_hidden_neurons, n_classes, learning_rate, n_epochs)
    model.load_state_dict(torch.load(dir_path+'\SVM_model'))
        

# -----------------------------------------------------------------------------------------------
# Input selected microphone recording to determine drone
# -----------------------------------------------------------------------------------------------
test_data = model.predict(readfile())
test_labels = [0, 1, 2, 3, 4]

# -----------------------------------------------------------------------------------------------
# Formatting output of classifier
# -----------------------------------------------------------------------------------------------
item_lst = ['Altura.csv', 'ANWB.csv', 'ATMOS.csv', 'Autel_Evo.csv', 'Phantom.csv']
val, idx = torch.mode(test_data)
print('The detected drone is: ', item_lst[int(val)])

"""
for i in test_labels:
    if i == test_data.item():
        print('The detected drone is: ', item_lst[i])
    else:
        print('The file provided does not match any of the drones in the database.')
"""