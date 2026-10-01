"""
Telecom Paris : projet Clustering 1D avec Méthode Seuil (abandonné)

Projet réalisé par :
- Samy Bali
- Hussein El Gouch 
- Issa Daou
- Anas El Barhdadi

Encadré par Jean-Louis Dessalles

Ce fichier implémente un clustering 1D avec:
- Génération de données gaussiennes
- Conversion en code Gray
- Évaluation par seuil
"""

import numpy as np
import random
import csv

class ClusteringGray:
    def __init__(self, centers, stds, points_per_center=50, bits=8, threshold=2, seed=42):
        """
        Initialise le clustering incrémental avec des paramètres donnés.
        centers : liste d'entiers (ex. [10, 20, 35])
        stds : liste de std correspondantes (ex. [2, 2, 3])
        points_per_center : nb de points par centre
        bits : nombre de bits pour le code Gray
        threshold : distance max en Hamming pour être dans le même cluster
        seed : graine aléatoire
        """
        self.centers = centers
        self.stds = stds
        self.points_per_center = points_per_center
        self.bits = bits
        self.threshold = threshold
        self.seed = seed
        self.data = self.generate_data()
        self.clusters = []

    def generate_data(self):
        """
        Génère un dataset 1D composé de gaussiennes autour de centres (entiers).
        Retourne un tableau numpy des valeurs flottantes.
        """
        np.random.seed(self.seed)
        data = []
        for c, s in zip(self.centers, self.stds):
            vals = np.random.normal(c, s, self.points_per_center)
            data.extend(vals)
        random.shuffle(data)
        return np.array(data)

    @staticmethod
    def decimal_to_gray(n, bits=8):
        """
        Convertit un entier n >= 0 en code de Gray (string de '0'/'1') sur 'bits' bits.
        """
        g = n ^ (n >> 1)
        return format(g, f'0{bits}b')

    @staticmethod
    def float_to_int_gray(value, bits=8):
        """
        Arrondit la valeur (float) -> int, puis code de Gray.
        """
        n = int(round(value))
        if n < 0: 
            n = 0
        max_val = (1 << bits) - 1
        if n > max_val:
            n = max_val
        return ClusteringGray.decimal_to_gray(n, bits=bits)

    @staticmethod
    def hamming_distance(b1, b2):
        """
        Calcule la distance de Hamming entre deux chaînes binaires.
        """
        return sum(c1 != c2 for c1, c2 in zip(b1, b2))

    def update_prototype_bitwise_majority(self, members):
        """
        Calcule le prototype par majority sur chaque bit, 
        members = liste (val, bitstring)
        """
        count_ones = [0]*self.bits
        for (v, bstr) in members:
            for i, bit in enumerate(bstr):
                if bit == '1':
                    count_ones[i] += 1

        new_proto = []
        nb = len(members)
        for i in range(self.bits):
            if count_ones[i] >= (nb/2):
                new_proto.append('1')
            else:
                new_proto.append('0')
        return ''.join(new_proto)

    def incremental_clustering_gray(self):
        """
        Effectue un clustering incrémental sur 'data' (floats),
        - Convertit en code de Gray (bits=8 par défaut)
        - threshold : distance max en Hamming pour être dans le même cluster
        Retourne une liste de clusters.
        """
        for val in self.data:
            # 1) conversion en code Gray
            b = ClusteringGray.float_to_int_gray(val, bits=self.bits)

            if not self.clusters:
                # premier cluster
                self.clusters.append({
                    'prototype': b,
                    'members': [(val, b)]
                })
                continue

            # chercher le cluster le plus proche
            best_cluster_idx = None
            best_dist = None

            for i, c in enumerate(self.clusters):
                dist = ClusteringGray.hamming_distance(b, c['prototype'])
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_cluster_idx = i

            # décider rattachement
            if best_dist <= self.threshold:
                self.clusters[best_cluster_idx]['members'].append((val, b))
                # mise à jour prototype (bit-wise majority)
                self.clusters[best_cluster_idx]['prototype'] = self.update_prototype_bitwise_majority(
                    self.clusters[best_cluster_idx]['members']
                )
            else:
                # nouveau cluster
                self.clusters.append({
                    'prototype': b,
                    'members': [(val, b)]
                })

    def save_to_csv(self, filename="dataset.csv"):
        """
        Sauvegarde les données générées dans un fichier CSV.
        """
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["value"])
            for val in self.data:
                writer.writerow([val])

    def print_clusters(self):
        """
        Affiche les clusters trouvés ssss.
        """
        print(f"Nombre de clusters trouvés : {len(self.clusters)}")
        for i, c in enumerate(self.clusters):
            vals = [m[0] for m in c['members']]
            print(f"\nCluster {i}: size={len(vals)}")
            print(f"Prototype (Gray) = {c['prototype']}")
            print(f"Quelques valeurs = {vals[:10]} ...")


# 5) EXEMPLE D'UTILISATION
if __name__ == "__main__":
    # 1) Création de l'objet de clustering
    centers = [10, 20, 35]
    stds = [2, 2, 3]
    clustering = ClusteringGray(centers, stds, points_per_center=50, bits=8, threshold=2, seed=42)

    # 2) Sauvegarde en CSV
    clustering.save_to_csv("dataset.csv")

    # 3) Clustering
    clustering.incremental_clustering_gray()

    # 4) Affichage des clusters
    clustering.print_clusters()
