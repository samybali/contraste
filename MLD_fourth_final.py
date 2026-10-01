"""
Telecom Paris : projet Contraste

Projet réalisé par : Samy Bali, Hussein El gouch, Issa Daou, Anas El Barhdadi

Encadré par Jean-Louis Dessalles
"""

import math
import random
from math import log2
import numpy as np

# --------------------
# Classe Cluster
# --------------------
class Cluster:
    def __init__(self, prefix_bits, total_bits):
        """
        Crée un cluster représentant un prototype.
        :param prefix_bits: partie fixe du motif, sans jokers.
        :param total_bits: longueur totale des motifs (ici 8 bits).
        """
        self.prefix_bits = prefix_bits   # bits fixes du prototype
        self.total_bits = total_bits     # longueur totale (8 bits dans cet exemple)
        self.size = 0                    # nombre d'instances affectées au cluster
    
    def prefix_length(self):
        """Retourne le nombre de bits fixes (longueur du préfixe)."""
        return len(self.prefix_bits)
    
    def wildcards_count(self):
        """Retourne le nombre de jokers ('*') dans le prototype."""
        return self.total_bits - self.prefix_length()
    
    def pattern_str(self):
        """Retourne le motif complet (préfixe suivi des jokers)."""
        return self.prefix_bits + "*" * self.wildcards_count()
    
    def common_prefix_with_instance(self, instance):
        """
        Calcule la longueur du préfixe commun entre ce cluster et l'instance.
        """
        cp = 0
        for i in range(min(self.prefix_length(), len(instance))):
            if instance[i] == self.prefix_bits[i]:
                cp += 1
            else:
                break
        return cp

# --------------------
# Calcul du coût MDL
# --------------------
def calculate_total_cost(clusters):
    """
    Calcule le coût MDL total = coût du modèle + coût des données + coût d'assignation.
      - Coût du modèle : somme des longueurs de préfixe (bits fixes) de chaque cluster.
      - Coût des données : pour chaque instance, le nombre de bits non couverts (jokers).
      - Coût d'assignation : ici, on estime un coût basé sur la distribution des instances.
    """
    total_instances = sum(c.size for c in clusters)
    if total_instances == 0:
        return 0.0

    # Coût du modèle : somme des bits fixes de tous les clusters.
    model_cost = sum(c.prefix_length() for c in clusters)
    # Coût des données : chaque instance doit coder les bits manquants (les jokers).
    data_cost = sum(c.size * c.wildcards_count() for c in clusters)
    # Coût d'assignation (schématique ici) : on fait varier selon la distribution des instances.
    assign_cost = 0.0
    for c in clusters:
        if c.size > 0:
            assign_cost += c.size * (log2(total_instances) - log2(c.size))
    
    return model_cost + data_cost + assign_cost

# --------------------
# Fonction d'insertion avec limite de 3 jokers maximum
# --------------------
def inserer_instance(instance, clusters, max_wildcards=3):
    """
    Insère une instance (chaîne binaire) dans la liste de clusters en respectant :
      - Une généralisation suffixée : dès le premier bit de divergence, tous les bits à droite sont remplacés par '*'.
      - La généralisation ne doit pas excéder max_wildcards (3 jokers maximum).
    
    Retourne un tuple indiquant l'option choisie ("attach" ou "new") et le coût MDL associé.
    """
    # Option "nouveau" : créer un cluster avec le motif identique à l'instance
    nouveau_cluster = Cluster(instance, len(instance))
    nouveau_cluster.size = 1
    cout_nouveau = calculate_total_cost(clusters + [nouveau_cluster])
    meilleur_cout = cout_nouveau
    best_option = ("new", None, instance)  # (option, index du cluster si rattachement, nouveau_prefix)
    
    # Tester le rattachement de l'instance à chacun des clusters existants
    for i, cluster in enumerate(clusters):
        # Calculer le préfixe commun entre le prototype (cluster) et l'instance
        cp = cluster.common_prefix_with_instance(instance)
        # Si cp est inférieur à la longueur actuelle du préfixe, une généralisation est nécessaire
        if cp < cluster.prefix_length():
            # Les jokers que nous aurions à ajouter correspondent au nombre de bits non communs de l'instance
            wildcards_needed = len(instance) - cp
            # Si on dépasse le seuil max de 3 jokers, on écarte cette option
            if wildcards_needed > max_wildcards:
                continue
            new_prefix = cluster.prefix_bits[:cp]  # nouveau préfixe réduit à cp bits
        else:
            # Le cluster couvre déjà complètement l'instance (aucune généralisation nécessaire)
            new_prefix = cluster.prefix_bits
        
        # Simuler la mise à jour temporaire du cluster pour évaluer le coût
        saved_prefix = cluster.prefix_bits
        saved_size = cluster.size
        
        cluster.prefix_bits = new_prefix
        cluster.size += 1  # simuler l'ajout de l'instance
        
        cout_attach = calculate_total_cost(clusters)
        
        # Restaurer l'état initial du cluster
        cluster.prefix_bits = saved_prefix
        cluster.size = saved_size
        
        if cout_attach < meilleur_cout:
            meilleur_cout = cout_attach
            best_option = ("attach", i, new_prefix)
    
    # Appliquer l'option optimale trouvée
    if best_option[0] == "new":
        clusters.append(nouveau_cluster)
        print(f"Nouveau prototype créé: {nouveau_cluster.pattern_str()} pour l'instance {instance}")
    else:
        i = best_option[1]
        new_prefix = best_option[2]
        if new_prefix != clusters[i].prefix_bits:
            old_pattern = clusters[i].pattern_str()
            new_pattern = new_prefix + "*" * (len(instance) - len(new_prefix))
            print(f"Prototype généralisé de {old_pattern} vers {new_pattern} pour l'instance {instance}")
        clusters[i].prefix_bits = new_prefix
        clusters[i].size += 1
    
    return best_option, meilleur_cout
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
    
donnees = []

for i in range(1000):
    if i % 3 == 0:
        donnees.append(float_to_int_gray(np.random.normal(100, 3),8))
    elif i % 3 == 1:
        donnees.append(float_to_int_gray(np.random.normal(150, 4),8))
    elif i % 3 == 2:
        donnees.append(float_to_int_gray(np.random.normal(50, 5),8))

donnees_bin_20 = list(set(donnees))
# --------------------
# Exemple de test : 3 clusters de nombres proches en 8 bits
# --------------------
# Définition manuelle de 3 groupes (clusters) de nombres proches (en décimal)
# Cluster 1 : Autour de 50
cluster1 = ['00110010', '00110100', '00110000', '00110011', '00110001', '00110101', '00101111']
# Cluster 2 : Autour de 150
cluster2 = ['10010110', '10010100', '10010111', '10010101', '10010011', '10011000', '10010110', '10011001']
# Cluster 3 : Autour de 200
cluster3 = ['11001000', '11001010', '11001001', '11001011', '11000101', '11001100', '11001110', '11001111']

# Fusionner les listes pour simuler une arrivée incrémentale mélangée
donnees_8bit = cluster1 + cluster2 + cluster3
random.shuffle(donnees_8bit)

clusters = []   # liste des clusters/prototypes
log_entries = []

# print("Instances générées (8 bits):")
# print(donnees_8bit, "\n")

print("Instances générées (20 bits):")
print(donnees_bin_20, "\n")

# Insertion incrémentale de chaque instance
for instance in donnees_bin_20:
    opt, cost = inserer_instance(instance, clusters, max_wildcards=4)
    log_entries.append((instance, opt, cost))
    print(f"Insertion instance {instance} -> Option: {opt}, Coût MDL: {cost:.2f}")

print("\nPrototypes finaux:")
for cluster in clusters:
    print(f" - {cluster.pattern_str()} (nombre d'instances = {cluster.size})")
