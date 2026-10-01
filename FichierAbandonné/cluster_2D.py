"""
Telecom Paris : projet Clustering 1D avec Méthode Seuil (abandonné)

Projet réalisé par :
- Samy Bali
- Hussein El Gouch 
- Issa Daou
- Anas El Barhdadi

Encadré par Jean-Louis Dessalles

Ce fichier implémente un clustering 2D avec:
- Génération de données gaussiennes
- Conversion en code Gray
- Évaluation par seuil
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import csv
from clustering_1D import ClusteringGray

class Cluster2D:
    
    def __init__(self, centers,std,points_per_centre=50, bits=8, threshold=4,seed=42):
        """
        Génère un dataset 2D composé de gaussiennes autour de centres (entiers).
        centers : liste de tuples (ex. [(10, 10), (20, 20), (35, 35)])
        """
        self.centers = centers
        self.std = std
        self.points_per_centre = points_per_centre
        self.bits = bits
        self.threshold = threshold
        self.seed = seed
        self.clusters=[]
        
    def generate_data_2D(self):
        np.random.seed(self.seed)
        data = []
        for c, s in zip(self.centers, self.std):
            vals = np.random.normal(c, s, (self.points_per_centre, 2))
            data.extend(vals)
        
        # on mélange
        random.shuffle(data)
        # Enregistrement des données dans un fichier CSV
        with open('data_2D.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['X', 'Y'])
            writer.writerows(data)
        return np.array(data)
    
    def code_gray_2D(self,point):
        """
        Convertit les coordonnées x et y d'un point en code de Gray et retourne la concaténation
        des deux codes.
        """
        x,y = point
        x_gray = ClusteringGray.float_to_int_gray(x, self.bits)
        y_gray = ClusteringGray.float_to_int_gray(y, self.bits)
        
        return x_gray + y_gray
    
    def incremental_clustering_2D(self):
        """
        Réalise un clustering 2D incrémental avec la méthode du seuil.
        """
        data= self.generate_data_2D()
        for point in data:
            b= self.code_gray_2D(point)
            
            if not self.clusters:
                self.clusters.append({"prototype":b, "members": [(point,b)]})
                continue
            # On cherche le cluster le plus proche
            best_cluster_idx = None
            best_dist= None
            for i, c in enumerate(self.clusters):
                dist = ClusteringGray.hamming_distance(b, c['prototype'])
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_cluster_idx = i
                    
            # décider rattachement
            if best_dist <= self.threshold:
                self.clusters[best_cluster_idx]['members'].append((point, b))
                # mise à jour prototype (bit-wise majority)
                self.clusters[best_cluster_idx]['prototype'] = ClusteringGray.update_prototype_bitwise_majority(
                    self.clusters[best_cluster_idx]['members']
                )
            else:
                # nouveau cluster
                self.clusters.append({
                    'prototype': b,
                    'members': [(point, b)]
                })
                
            def save_to_csv(self, filename="dataset_2D.csv"):
                """
                Sauvegarde les données générées dans un fichier CSV.
                """
                with open(filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["X", "Y"])
                    for point in self.data:
                        writer.writerow([point[0],point[1]])
                        
            def plot_clusters(self):
                """
                Affiche les clusters dans un graphique.
                """
                for cluster in self.clusters:
                    x = [point[0] for point in cluster['members']]
                    y = [point[1] for point in cluster['members']]
                    plt.scatter(x, y)
                plt.show()
                
# Exemple d'utilisation
centers = [(15, 15), (20, 20), (35, 35)]
std = [2, 2, 2]
cluster = Cluster2D(centers,std)
cluster.incremental_clustering_2D()
cluster.save_to_csv()
cluster.plot_clusters()


            
            
        


