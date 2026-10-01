"""
Telecom Paris : projet Clustering 2D avec Gray Code et Méthode Incrémentale

Projet réalisé par :
- Samy Bali
- Hussein El Gouch 
- Issa Daou
- Anas El Barhdadi

Encadré par Jean-Louis Dessalles

Ce fichier implémente un clustering 2D incrémental qui résout le problème 
de conflit entre les dimensions x et y comme mentionné dans le suivi.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.colors import hsv_to_rgb
import csv

class Clustering2D:
    def __init__(self, bits=20, threshold_x=2, threshold_y=2, seed=42):
        """
        Initialise le clustering incrémental 2D.
        
        Args:
            bits: nombre de bits pour le code Gray (par défaut: 20 bits)
            threshold_x: seuil de distance pour la dimension x
            threshold_y: seuil de distance pour la dimension y
            seed: graine aléatoire
        """
        self.bits = bits
        self.threshold_x = threshold_x
        self.threshold_y = threshold_y
        self.seed = seed
        self.clusters = []  # Liste pour stocker les clusters
        self.data = []      # Liste pour stocker les données
        
    def generate_random_data(self, n_points=30, centers=[(10, 10), (25, 25), (10, 25)], stds=[(2, 2), (3, 3), (2, 3)]):
        """
        Génère des données 2D aléatoires avec des distributions gaussiennes.
        
        Args:
            n_points: nombre total de points à générer
            centers: centres des distributions gaussiennes (x, y)
            stds: écarts-types des distributions gaussiennes (x, y)
        
        Returns:
            Tableau numpy de points 2D
        """
        np.random.seed(self.seed)
        data = []
        
        # Distribution équitable des points parmi les centres
        points_per_center = n_points // len(centers)
        remainder = n_points % len(centers)
        
        for i, (center, std) in enumerate(zip(centers, stds)):
            n = points_per_center + (1 if i < remainder else 0)
            points = np.random.normal(center, std, (n, 2))
            data.extend(points)
        
        # Mélange des points
        random.seed(self.seed)
        random.shuffle(data)
        
        self.data = np.array(data)
        return self.data
    
    @staticmethod
    def decimal_to_gray(n, bits=8):
        """
        Convertit un nombre décimal en code Gray.
        
        Args:
            n: nombre entier à convertir
            bits: nombre de bits pour la représentation
            
        Returns:
            Chaîne binaire représentant le code Gray
        """
        # Conversion vers code Gray
        g = n ^ (n >> 1)
        return format(g, f'0{bits}b')
    
    @staticmethod
    def float_to_int_gray(value, bits=20):
        """
        Convertit une valeur flottante en code Gray en maximisant la précision.
        
        La méthode utilise un décalage adaptatif de la virgule pour représenter 
        le maximum d'information dans les bits disponibles:
        - Si le nombre est très grand, tous les bits seront utilisés pour la partie entière
        - Si le nombre est très petit, la plupart des bits seront utilisés pour la partie décimale
        
        Args:
            value: valeur flottante à convertir
            bits: nombre total de bits pour la représentation (défaut: 20)
            
        Returns:
            Chaîne binaire représentant le code Gray
        """
        # Valeur absolue pour simplifier le traitement
        abs_value = abs(value)
        
        # Déterminer le nombre de bits nécessaires pour la partie entière
        if abs_value < 1:
            # Cas spécial: nombre inférieur à 1
            integer_bits = 1  # Un bit pour le zéro
        else:
            # Calculer le nombre de bits nécessaires pour la partie entière
            integer_bits = max(1, int(np.log2(abs_value)) + 2)  # +1 pour arrondir, +1 pour le bit de signe
        
        # Limiter à la taille maximale disponible
        integer_bits = min(integer_bits, bits)
        
        # Déterminer le facteur de mise à l'échelle pour maximiser la précision
        decimal_bits = bits - integer_bits
        scale_factor = 2 ** decimal_bits
        
        # Mettre à l'échelle la valeur pour utiliser tous les bits disponibles
        scaled_value = int(round(value * scale_factor))
        
        # S'assurer que la valeur est dans les limites représentables
        max_value = (1 << bits) - 1
        min_value = -(1 << bits)
        scaled_value = max(min_value, min(scaled_value, max_value))
        
        # Conversion en binaire avec gestion du signe
        if scaled_value < 0:
            # Pour les nombres négatifs, utiliser le complément à 2
            binary_value = format((1 << bits) + scaled_value, f'0{bits}b')[-bits:]
        else:
            binary_value = format(scaled_value, f'0{bits}b')[-bits:]
        
        # Appliquer la conversion en code Gray
        n = int(binary_value, 2)
        g = n ^ (n >> 1)
        return format(g, f'0{bits}b')

    @staticmethod
    def hamming_distance(b1, b2):
        """
        Calcule la distance de Hamming entre deux chaînes binaires,
        en ignorant les positions avec '*' (considérées comme des jokers).
        
        Args:
            b1: première chaîne binaire (peut contenir des '*')
            b2: deuxième chaîne binaire (peut contenir des '*')
            
        Returns:
            Distance de Hamming (nombre de bits différents, en ignorant les '*')
        """
        distance = 0
        for c1, c2 in zip(b1, b2):
            # Ignorer les positions où l'un des deux caractères est '*'
            if c1 == '*' or c2 == '*':
                continue
            # Sinon, compter les différences
            if c1 != c2:
                distance += 1
        return distance

    @staticmethod
    def weighted_hamming_distance(b1, b2):
        """
        Calcule la distance de Hamming pondérée entre deux chaînes binaires,
        où chaque bit a un poids de 2^position en partant de la gauche.
        Les positions avec '*' sont ignorées (détails).
        
        Args:
            b1: première chaîne binaire (peut contenir des '*')
            b2: deuxième chaîne binaire (peut contenir des '*')
            
        Returns:
            Distance de Hamming pondérée
        """
        distance = 0
        n = len(b1)
        
        for i, (c1, c2) in enumerate(zip(b1, b2)):
            # Ignorer les positions avec '*'
            if c1 == '*' or c2 == '*':
                continue
            # Si différents, ajouter la distance pondérée (2^poids)
            if c1 != c2:
                # Plus le bit est à gauche, plus son poids est important
                weight = 2 ** (n - i - 1)
                distance += weight
                
        return distance

    @staticmethod
    def euclidean_distance(p1, p2):
        """
        Calcule la distance euclidienne entre deux points dans l'espace.
        
        Args:
            p1: premier point (coordonnées x, y)
            p2: deuxième point (coordonnées x, y)
            
        Returns:
            Distance euclidienne entre les deux points
        """
        return np.sqrt(np.sum((np.array(p1) - np.array(p2))**2))
    
    def update_prototype(self, members, dimension, significant_bits=4):
        """
        Calcule un nouveau prototype en conservant exactement 4 bits significatifs.
        Les autres bits sont considérés comme des détails et remplacés par des '*'.
        
        Args:
            members: liste de membres (points avec leur représentation binaire)
            dimension: 0 pour x, 1 pour y
            significant_bits: nombre de bits significatifs à conserver (par défaut 4)
            
        Returns:
            Nouveau prototype (chaîne binaire avec '*' pour les bits non significatifs)
        """
        count_ones = [0] * self.bits
        
        # Compter les '1' à chaque position
        for member in members:
            bin_rep = member[1][dimension]  # Représentation binaire de la dimension
            for i, bit in enumerate(bin_rep):
                if bit == '1':
                    count_ones[i] += 1
        
        # Construire le nouveau prototype par vote majoritaire
        new_proto = []
        nb = len(members)
        
        for i in range(self.bits):
            # Ne garder que les bits significatifs (de poids fort)
            if i < significant_bits:
                if count_ones[i] >= nb/2:
                    new_proto.append('1')
                else:
                    new_proto.append('0')
            else:
                # Remplacer les bits de détails par '*'
                new_proto.append('*')
                
        return ''.join(new_proto)
    
    def add_point(self, point):
        """
        Ajoute un point au clustering de manière incrémentale en utilisant la distance euclidienne.
        
        Args:
            point: point 2D à ajouter
        """
        # Conversion en codes Gray (toujours utile pour l'affichage)
        x_gray = self.float_to_int_gray(point[0], self.bits)
        y_gray = self.float_to_int_gray(point[1], self.bits)
        
        if not self.clusters:
            # Premier cluster
            self.clusters.append({
                'prototype_x': x_gray,
                'prototype_y': y_gray,
                'center_x': point[0],
                'center_y': point[1],
                'members': [(point, (x_gray, y_gray))]
            })
            return
        
        # Chercher le cluster le plus proche en utilisant la distance euclidienne
        best_idx = None
        min_distance = float('inf')
        
        for i, cluster in enumerate(self.clusters):
            # Calculer la distance euclidienne entre le point et le centre du cluster
            center = (cluster['center_x'], cluster['center_y'])
            dist = self.euclidean_distance(point, center)
            
            if dist < min_distance:
                min_distance = dist
                best_idx = i
        
        # Seuil adapté pour la distance euclidienne, à ajuster selon les données
        euclidean_threshold = 50  # Valeur plus grande pour accommoder l'échelle des coordonnées
        
        # Décision d'assignation
        if min_distance <= euclidean_threshold:
            # Ajout au cluster le plus proche
            self.clusters[best_idx]['members'].append((point, (x_gray, y_gray)))
            
            # Mise à jour des prototypes et centres
            self.clusters[best_idx]['prototype_x'] = self.update_prototype(self.clusters[best_idx]['members'], 0)
            self.clusters[best_idx]['prototype_y'] = self.update_prototype(self.clusters[best_idx]['members'], 1)
            
            # Mise à jour du centre (moyenne des points)
            points = np.array([m[0] for m in self.clusters[best_idx]['members']])
            self.clusters[best_idx]['center_x'] = np.mean(points[:, 0])
            self.clusters[best_idx]['center_y'] = np.mean(points[:, 1])
        else:
            # Nouveau cluster - le point est trop éloigné des clusters existants
            self.clusters.append({
                'prototype_x': x_gray,
                'prototype_y': y_gray,
                'center_x': point[0],
                'center_y': point[1],
                'members': [(point, (x_gray, y_gray))]
            })

    def add_point_weighted_hamming(self, point):
        """
        Ajoute un point au clustering en utilisant la distance de Hamming pondérée.
        
        Args:
            point: point 2D à ajouter
        """
        # Conversion en codes Gray
        x_gray = self.float_to_int_gray(point[0], self.bits)
        y_gray = self.float_to_int_gray(point[1], self.bits)
        
        if not self.clusters:
            # Premier cluster
            self.clusters.append({
                'prototype_x': x_gray,
                'prototype_y': y_gray,
                'center_x': point[0],
                'center_y': point[1],
                'members': [(point, (x_gray, y_gray))]
            })
            return
        
        # Chercher le cluster le plus proche en utilisant la distance de Hamming pondérée
        best_x_idx = None
        best_x_dist = float('inf')
        best_y_idx = None
        best_y_dist = float('inf')
        
        for i, cluster in enumerate(self.clusters):
            dist_x = self.weighted_hamming_distance(x_gray, cluster['prototype_x'])
            if dist_x < best_x_dist:
                best_x_dist = dist_x
                best_x_idx = i
                
            dist_y = self.weighted_hamming_distance(y_gray, cluster['prototype_y'])
            if dist_y < best_y_dist:
                best_y_dist = dist_y
                best_y_idx = i
        
        # Seuils adaptés pour la distance de Hamming pondérée avec 20 bits
        weighted_threshold_x = 2 ** (self.bits - 6)  # Pour 20 bits, accepte différences dans les bits moins significatifs
        weighted_threshold_y = 2 ** (self.bits - 6)
        
        # Décision d'assignation
        if best_x_dist <= weighted_threshold_x and best_y_dist <= weighted_threshold_y:
            # Si les deux dimensions sont proches, on regarde lequel des deux est le plus proche
            if best_x_idx == best_y_idx:
                # Même cluster optimal pour les deux dimensions
                best_idx = best_x_idx
            else:
                # Résoudre le conflit par la méthode décrite dans le suivi
                # 1. Pour le cluster optimal en x: fixer x et calculer l'écart en y
                weight_x = abs(point[1] - self.clusters[best_x_idx]['center_y'])
                # 2. Pour le cluster optimal en y: fixer y et calculer l'écart en x
                weight_y = abs(point[0] - self.clusters[best_y_idx]['center_x'])
                
                # 3. Sélectionner le cluster avec le poids minimal
                if weight_x < weight_y:
                    best_idx = best_x_idx
                else:
                    best_idx = best_y_idx
                    
            # Ajout au cluster choisi
            self.clusters[best_idx]['members'].append((point, (x_gray, y_gray)))
            
            # Mise à jour des prototypes et centres
            self.clusters[best_idx]['prototype_x'] = self.update_prototype(self.clusters[best_idx]['members'], 0)
            self.clusters[best_idx]['prototype_y'] = self.update_prototype(self.clusters[best_idx]['members'], 1)
            
            # Mise à jour du centre (moyenne des points)
            points = np.array([m[0] for m in self.clusters[best_idx]['members']])
            self.clusters[best_idx]['center_x'] = np.mean(points[:, 0])
            self.clusters[best_idx]['center_y'] = np.mean(points[:, 1])
        else:
            # Nouveau cluster - le point est trop éloigné des clusters existants
            self.clusters.append({
                'prototype_x': x_gray,
                'prototype_y': y_gray,
                'center_x': point[0],
                'center_y': point[1],
                'members': [(point, (x_gray, y_gray))]
            })
    
    def add_point_composant_based(self, point):
        """
        Ajoute un point au clustering en traitant séparément les composantes x et y,
        puis en résolvant les conflits selon les bits de poids fort.
        
        Cette approche considère que chaque composante (x,y) peut déjà appartenir à un 
        clustering en 1D et possède donc un prototype idéal en x et en y.
        
        Args:
            point: point 2D à ajouter
        """
        # Conversion en codes Gray
        x_gray = self.float_to_int_gray(point[0], self.bits)
        y_gray = self.float_to_int_gray(point[1], self.bits)
        
        if not self.clusters:
            # Premier point: création du premier cluster
            self.clusters.append({
                'prototype_x': x_gray,
                'prototype_y': y_gray,
                'center_x': point[0],
                'center_y': point[1],
                'members': [(point, (x_gray, y_gray))]
            })
            return
        
        # Vérifier pour chaque cluster existant la ressemblance des prototypes
        # en se concentrant sur les bits de poids fort (significatifs)
        match_x_idx = []  # Indices des clusters dont le prototype x correspond
        match_y_idx = []  # Indices des clusters dont le prototype y correspond
        
        for i, cluster in enumerate(self.clusters):
            # Calcul de la distance entre les prototypes, en mettant l'accent sur les bits significatifs
            dist_x = self.weighted_hamming_distance(x_gray, cluster['prototype_x'])
            dist_y = self.weighted_hamming_distance(y_gray, cluster['prototype_y'])
            
            # Seuils pour déterminer si les prototypes sont similaires
            # Ces seuils sont à ajuster en fonction de vos données
            weight_threshold_x = 2 ** (self.bits - 3)  # Pour 8 bits, accepte différences dans les 3 bits les moins significatifs
            weight_threshold_y = 2 ** (self.bits - 3)
            
            if dist_x <= weight_threshold_x:
                match_x_idx.append(i)
            
            if dist_y <= weight_threshold_y:
                match_y_idx.append(i)
        
        # Cas 1: Aucune correspondance - créer un nouveau cluster
        if not match_x_idx and not match_y_idx:
            self.clusters.append({
                'prototype_x': x_gray,
                'prototype_y': y_gray,
                'center_x': point[0],
                'center_y': point[1],
                'members': [(point, (x_gray, y_gray))]
            })
            return
        
        # Cas 2: Correspondance pour x seulement
        if match_x_idx and not match_y_idx:
            # Choisir le cluster avec la meilleure correspondance en x
            # best_x_idx = min(match_x_idx, key=lambda i: self.weighted_hamming_distance(x_gray, self.clusters[i]['prototype_x']))
            # self._add_to_cluster(best_x_idx, point, x_gray, y_gray)
            self.clusters.append({
                'prototype_x': x_gray,
                'prototype_y': y_gray,
                'center_x': point[0],
                'center_y': point[1],
                'members': [(point, (x_gray, y_gray))]
            })
            return
        
        # Cas 3: Correspondance pour y seulement
        if not match_x_idx and match_y_idx:
            # Choisir le cluster avec la meilleure correspondance en y
            # best_y_idx = min(match_y_idx, key=lambda i: self.weighted_hamming_distance(y_gray, self.clusters[i]['prototype_y']))
            # self._add_to_cluster(best_y_idx, point, x_gray, y_gray)
            self.clusters.append({
                'prototype_x': x_gray,
                'prototype_y': y_gray,
                'center_x': point[0],
                'center_y': point[1],
                'members': [(point, (x_gray, y_gray))]
            })
            return
        
        # Cas 4: Correspondance pour x et y - vérifier s'il existe un cluster commun
        common_clusters = set(match_x_idx).intersection(set(match_y_idx))
        if common_clusters:
            # S'il y a un ou plusieurs clusters qui correspondent à la fois à x et y,
            # choisir celui avec la meilleure correspondance combinée
            best_cluster_idx = min(common_clusters, 
                                   key=lambda i: (self.weighted_hamming_distance(x_gray, self.clusters[i]['prototype_x']) + 
                                                 self.weighted_hamming_distance(y_gray, self.clusters[i]['prototype_y'])))
            self._add_to_cluster(best_cluster_idx, point, x_gray, y_gray)
            return
        
        # Cas 5: Conflit - x et y correspondent à des clusters différents
        # Appliquer la méthode de pondération décrite dans le suivi
        
        # Trouver le meilleur cluster pour x et le meilleur pour y
        best_x_idx = min(match_x_idx, key=lambda i: self.weighted_hamming_distance(x_gray, self.clusters[i]['prototype_x']))
        best_y_idx = min(match_y_idx, key=lambda i: self.weighted_hamming_distance(y_gray, self.clusters[i]['prototype_y']))
        
        # 1. Pour le cluster optimal en x: fixer x et calculer l'écart avec sa valeur y
        weight_x = abs(point[1] - self.clusters[best_x_idx]['center_y'])
        
        # 2. Pour le cluster optimal en y: fixer y et calculer l'écart avec sa valeur x
        weight_y = abs(point[0] - self.clusters[best_y_idx]['center_x'])
        
        # 3. Sélectionner le cluster avec le poids minimal
        if weight_x <= weight_y:
            self._add_to_cluster(best_x_idx, point, x_gray, y_gray)
        else:
            self._add_to_cluster(best_y_idx, point, x_gray, y_gray)
    
    def _add_to_cluster(self, cluster_idx, point, x_gray, y_gray):
        """
        Méthode utilitaire pour ajouter un point à un cluster existant et mettre à jour les prototypes.
        
        Args:
            cluster_idx: indice du cluster auquel ajouter le point
            point: point 2D à ajouter
            x_gray: représentation en code Gray de la coordonnée x
            y_gray: représentation en code Gray de la coordonnée y
        """
        # Ajout au cluster choisi
        self.clusters[cluster_idx]['members'].append((point, (x_gray, y_gray)))
        
        # Mise à jour des prototypes (en préservant les bits de poids fort)
        self.clusters[cluster_idx]['prototype_x'] = self.update_prototype(self.clusters[cluster_idx]['members'], 0)
        self.clusters[cluster_idx]['prototype_y'] = self.update_prototype(self.clusters[cluster_idx]['members'], 1)
        
        # Mise à jour du centre (moyenne des points)
        points = np.array([m[0] for m in self.clusters[cluster_idx]['members']])
        self.clusters[cluster_idx]['center_x'] = np.mean(points[:, 0])
        self.clusters[cluster_idx]['center_y'] = np.mean(points[:, 1])
    
    def run_clustering(self):
        """
        Exécute le clustering sur toutes les données.
        """
        # Réinitialiser les clusters
        self.clusters = []
        # Ajouter les points un par un (approche incrémentale)
        for point in self.data:
            self.add_point(point)
    
    def run_weighted_hamming_clustering(self):
        """
        Exécute le clustering sur toutes les données avec la distance de Hamming pondérée.
        """
        # Réinitialiser les clusters
        self.clusters = []
        # Ajouter les points un par un (approche incrémentale)
        for point in self.data:
            self.add_point_weighted_hamming(point)

    def run_composant_based_clustering(self):
        """
        Exécute le clustering sur toutes les données avec l'approche basée sur les composantes.
        """
        # Réinitialiser les clusters
        self.clusters = []
        # Ajouter les points un par un (approche incrémentale)
        for point in self.data:
            self.add_point_composant_based(point)

    def visualize_clusters(self, title="Clustering 2D avec Gray Code", ax=None):
        """
        Visualise les clusters et leurs prototypes avec des couleurs aléatoires distinctes.
        
        Args:
            title: Titre du graphique
            ax: Axes matplotlib où tracer (si None, crée une nouvelle figure)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
            show_plot = True
        else:
            show_plot = False
            
        # Génération de couleurs aléatoires distinctes pour chaque cluster
        n_clusters = len(self.clusters)
        
        # Définir une graine aléatoire pour la reproductibilité
        np.random.seed(42)
        
        # Génération de couleurs distinctes en utilisant HSV
        # H (teinte) est réparti uniformément autour du cercle chromatique
        hues = np.random.permutation(np.linspace(0, 1, n_clusters+1))[:-1]
        # S (saturation) et V (valeur) sont maintenus élevés pour des couleurs vives
        saturations = 0.7 + 0.3 * np.random.random(n_clusters)
        values = 0.7 + 0.3 * np.random.random(n_clusters)
        
        # Conversion HSV -> RGB
        cluster_colors = [hsv_to_rgb((h, s, v)) for h, s, v in zip(hues, saturations, values)]
        
        # Tracer les points de chaque cluster
        for i, cluster in enumerate(self.clusters):
            points = np.array([m[0] for m in cluster['members']])
            color = cluster_colors[i]  # Couleur unique pour chaque cluster
            ax.scatter(points[:, 0], points[:, 1], color=color, alpha=0.7, label=f"Cluster {i+1}")
            
            # Tracer le centre du cluster
            ax.scatter(cluster['center_x'], cluster['center_y'], color=color, 
                     marker='X', s=100, edgecolors='black')
            # Annotations: binary prototypes
            ax.annotate(f"X: {cluster['prototype_x']}\nY: {cluster['prototype_y']}",
                       (cluster['center_x'], cluster['center_y']), 
                       xytext=(10, 5), textcoords='offset points',
                       bbox=dict(boxstyle="round", fc="w", alpha=0.8))
        
        ax.set_title(title)
        ax.set_xlabel("Dimension X")
        ax.set_ylabel("Dimension Y")
        ax.grid(alpha=0.3)
        ax.legend()
        
        if show_plot:
            plt.tight_layout()
            plt.show()
        
        return ax
        
    def save_to_csv(self, filename="clusters_2d.csv"):
        """
        Sauvegarde les résultats du clustering dans un fichier CSV.
        """
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Cluster', 'X', 'Y', 'Gray_Code_X', 'Gray_Code_Y'])
            for i, cluster in enumerate(self.clusters):
                for point, gray_codes in cluster['members']:
                    writer.writerow([i, point[0], point[1], gray_codes[0], gray_codes[1]])
        print(f"Données sauvegardées dans {filename}")

# Démonstration du clustering
if __name__ == "__main__":
    # Initialisation avec 20 bits
    clustering = Clustering2D(bits=20, threshold_x=2, threshold_y=2)
    
    # Génération de données avec des valeurs plus grandes pour utiliser les 20 bits
    data = clustering.generate_random_data(
        n_points=500,
        centers=[(50, 50), (100, 150), (50, 150), (150, 50)],  # Valeurs beaucoup plus grandes
        stds=[(8, 5), (8, 8), (5, 10), (8, 5)]  # Écarts-types adaptés
    )
    
    #Un second jeu de données avec des valeurs décimales et négatives
    # data_decimal = clustering.generate_random_data(
    #     n_points=15,
    #     centers=[(-50.75, 250.5), (75.25, -300.5), (100.125, 100.875)],  # Avec décimales et négatifs
    #     stds=[(5.5, 8.5), (8.5, 5.5), (10.25, 8.25)]
    # )
    
    # Combiner les deux jeux de données
    # clustering.data = np.vstack([data, data_decimal])
    
    # Exécuter les deux méthodes de clustering
    print("\nClustering avec distance euclidienne:")
    clustering.run_clustering()
    
    # Pour les tests, initialiser également le clustering Hamming avec 20 bits
    clustering_hamming = Clustering2D(bits=20, threshold_x=2, threshold_y=2)
    clustering_hamming.data = data
    clustering_hamming.run_weighted_hamming_clustering()
    
    # MÉTHODE 3: Clustering par composantes avec résolution de conflits
    print("\nClustering par composantes avec résolution de conflits:")
    clustering_composants = Clustering2D(bits=20, threshold_x=2, threshold_y=2)
    clustering_composants.data = data
    clustering_composants.run_composant_based_clustering()
    
    # Visualisation des résultats côte à côte (3 méthodes)
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    clustering.visualize_clusters(title="Distance Euclidienne", ax=axes[0])
    clustering_hamming.visualize_clusters(title="Distance Hamming Pondérée", ax=axes[1])
    clustering_composants.visualize_clusters(title="Clustering par Composantes", ax=axes[2])
    
    plt.tight_layout()
    plt.savefig("comparaison_trois_methodes.png", dpi=300)
    plt.show()
    
    # Affichage des informations sur les clusters
    print(f"\nRésultats - Méthode Euclidienne:")
    print(f"Nombre de clusters formés: {len(clustering.clusters)}")
    for i, cluster in enumerate(clustering.clusters):
        print(f"Cluster {i+1}: {len(cluster['members'])} points")
        print(f"  Prototype X: {cluster['prototype_x']}")
        print(f"  Prototype Y: {cluster['prototype_y']}")
        print(f"  Centre: ({cluster['center_x']:.2f}, {cluster['center_y']:.2f})")
    
    print(f"\nRésultats - Méthode Hamming Pondérée:")
    print(f"Nombre de clusters formés: {len(clustering_hamming.clusters)}")
    for i, cluster in enumerate(clustering_hamming.clusters):
        print(f"Cluster {i+1}: {len(cluster['members'])} points")
        print(f"  Prototype X: {cluster['prototype_x']}")
        print(f"  Prototype Y: {cluster['prototype_y']}")
        print(f"  Centre: ({cluster['center_x']:.2f}, {cluster['center_y']:.2f})")
    
    print(f"\nRésultats - Méthode par Composantes:")
    print(f"Nombre de clusters formés: {len(clustering_composants.clusters)}")
    for i, cluster in enumerate(clustering_composants.clusters):
        print(f"Cluster {i+1}: {len(cluster['members'])} points")
        print(f"  Prototype X: {cluster['prototype_x']}")
        print(f"  Prototype Y: {cluster['prototype_y']}")
        print(f"  Centre: ({cluster['center_x']:.2f}, {cluster['center_y']:.2f})")
    
    # Sauvegarde des résultats
    clustering.save_to_csv("clusters_euclidean.csv")
    clustering_hamming.save_to_csv("clusters_weighted_hamming.csv")
    clustering_composants.save_to_csv("clusters_composant_based.csv")
