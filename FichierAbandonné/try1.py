import numpy as np
import statistics as s

data = []

for i in range(1000):
    if i % 3 == 0:
        data.append(int(np.random.normal(100, 3)))
    elif i % 3 == 1:
        data.append(int(np.random.normal(3053, 4)))
    elif i % 3 == 2:
        data.append(int(np.random.normal(7079, 5)))

data = list(set(data))

data_bin = [bin(elt)[2:] for elt in data]
data_bin_20 = []
list_of_nearest_point = {}

def convert_to_20bits(bin_n):
    add = ""
    for _ in range(20-len(bin_n)):
        add += "0"
    return (add + bin_n)

def trouver_plus_proche(ent):
    data_bis = data.copy()
    data_bis.remove(ent)

    nearest = min(data_bis, key=lambda x: abs(x - ent))

    list_of_nearest_point[ent] = nearest

for elt in data:
    trouver_plus_proche(elt)

print(list_of_nearest_point)

def creer_proto(i1, i2):
    bin1 = convert_to_20bits(bin(i1)[2:])
    bin2 = convert_to_20bits(bin(i2)[2:])

    proto = ""

    for i in range(20):
        if bin1[i] == bin2[i]:
            proto += bin1[i]
        else:
            for i in range(20-i):
                proto += "*"
            break

    return proto

to_add = []
def ajouter_protos():
    new = []
    for elt in list_of_nearest_point:
        new.append(bin(elt)[2:])
        new.append(bin(list_of_nearest_point[elt])[2:])
        new.append(creer_proto(elt, list_of_nearest_point[elt]))
        to_add.append(creer_proto(elt, list_of_nearest_point[elt]))
    return new

new_data = ajouter_protos()

def hamming(bin1, bin2):
    
    return sum(b1 != b2 for b1, b2 in zip(bin1, bin2))

def compter_bits_commun(bin1, bin2):
    return bin(int(bin1, 2) & int(bin2, 2)).count('1')

while to_add != []:
    temp = {}
    nearest_points = {}
    for elt in to_add:
        a = elt.copy()
        temp[elt] = a.replace('*', '')
    for i, bin_num in enumerate(temp):
        max_common_bits = 0
        min_distance = float('inf')  
        most_common_bin = None
        
        
        for j, compare_bin in enumerate(temp):
            if i != j:  
                common_bits = compter_bits_commun(bin_num, compare_bin)
                distance = hamming(bin_num, compare_bin)
                
                
                if common_bits > max_common_bits or (common_bits == max_common_bits and distance < min_distance):
                    max_common_bits = common_bits
                    min_distance = distance
                    most_common_bin = compare_bin
        
        nearest_points[i] = bin_num





        
    




    

