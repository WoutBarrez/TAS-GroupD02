import csv
j = 0
with open("Altura.csv", "r") as file:
    csvreader = csv.reader(file)
    for row in csvreader:
        for i,e in enumerate(row) :
            row[i] = float(e)
            if j == 0 :
                print(row)
                j = 0