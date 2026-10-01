import math
from math import log2

"""
Implémentation détaillée du calcul de coût MDL (Minimum Description Length)
pour le clustering basé sur des prototypes binaires généralisés.

Le principe MDL vise à minimiser la somme de:
1. La complexité du modèle (les prototypes)
2. La complexité des données étant donné le modèle
3. Le coût d'assignation des instances aux prototypes
"""

class Cluster:
    def __init__(self, prefix_bits, total_bits):
        """
        Crée un cluster représentant un prototype.
        :param prefix_bits: partie fixe du motif, sans jokers.
        :param total_bits: longueur totale des motifs.
        """
        self.prefix_bits = prefix_bits   # bits fixes du prototype
        self.total_bits = total_bits     # longueur totale du motif
        self.size = 0                    # nombre d'instances dans le cluster
    
    def prefix_length(self):
        """Retourne le nombre de bits fixes (longueur du préfixe)."""
        return len(self.prefix_bits)
    
    def wildcards_count(self):
        """Retourne le nombre de jokers ('*') dans le prototype."""
        return self.total_bits - self.prefix_length()
    
    def pattern_str(self):
        """Retourne le motif complet (préfixe suivi des jokers)."""
        return self.prefix_bits + "*" * self.wildcards_count()

def calculate_model_cost(clusters):
    """
    Calcule le coût du modèle selon le principe MDL.
    
    'clusters' est une liste d'objets Cluster, où chaque cluster représente:
    - Un prototype de données caractérisé par un motif binaire partiellement généralisé
    - Un groupe d'instances similaires partageant des caractéristiques communes
    - Une abstraction permettant de compresser l'information des données
    
    Chaque élément de 'clusters' contient:
    - prefix_bits: chaîne de caractères avec les bits fixes du prototype (ex: "0110")
    - total_bits: longueur totale du motif incluant les jokers
    - size: nombre d'instances assignées à ce prototype
    
    Dans le contexte du MDL:
    - Plus il y a de clusters, plus le modèle est complexe (coût du modèle élevé)
    - Plus les prototypes ont de bits fixes, plus ils sont spécifiques et coûteux
    - L'ensemble des clusters forme le "modèle" qui décrit les données
    
    Exemple de 'clusters':
    [Cluster("0110", 8) avec 10 instances, Cluster("101", 8) avec 5 instances, ...]
    """
    return sum(c.prefix_length() for c in clusters)

def calculate_data_cost(clusters):
    """
    Calcule le coût des données étant donné le modèle.
    
    Pour chaque instance assignée à un prototype, nous devons coder:
    1. Les bits spécifiques qui correspondent aux positions des jokers dans le prototype
    
    Pour un prototype avec W jokers et N instances, le coût est N*W bits.
    Cela reflète le fait que chaque instance doit spécifier W bits d'information
    qui ne sont pas déjà capturés par le prototype.
    
    Exemple:
    - Prototype '01101***' avec 10 instances : coût = 10 * 3 = 30 bits
    - Prototype '010*****' avec 5 instances : coût = 5 * 5 = 25 bits
    """
    return sum(c.size * c.wildcards_count() for c in clusters)

def calculate_assignment_cost(clusters):
    """
    Calcule le coût d'assignation des instances aux prototypes.
    
    Ce coût représente l'information nécessaire pour spécifier à quel prototype
    chaque instance est assignée. Il est basé sur la théorie de l'information:
    
    - Si tous les prototypes sont également probables, le coût serait log2(M) par instance,
      où M est le nombre de prototypes.
    
    - En pratique, nous utilisons une codification optimale basée sur la distribution
      réelle des instances parmi les prototypes. Si P(c) est la probabilité qu'une instance
      appartienne au prototype c, alors le coût idéal est -log2(P(c)) bits par instance.
    
    Pour N_c instances dans le cluster c et N instances au total:
    P(c) = N_c / N
    Coût pour toutes les instances dans c = N_c * (-log2(N_c / N)) = N_c * (log2(N) - log2(N_c))
    """
    total_instances = sum(c.size for c in clusters)
    if total_instances == 0:
        return 0.0
    
    assign_cost = 0.0
    for c in clusters:
        if c.size > 0:
            # Coût Shannon-Fano/Huffman optimal pour coder l'assignation
            assign_cost += c.size * (log2(total_instances) - log2(c.size))
    
    return assign_cost

def calculate_total_mdl_cost(clusters):
    """
    Calcule le coût MDL total qui est la somme des trois composantes:
    1. Coût du modèle (complexité des prototypes)
    2. Coût des données (bits non capturés par les prototypes)
    3. Coût d'assignation (identification du prototype pour chaque instance)
    
    Le principe MDL vise à minimiser cette somme, réalisant un compromis optimal entre:
    - Un modèle trop simple (peu de prototypes très généraux) → coût du modèle bas mais coût des données élevé
    - Un modèle trop complexe (beaucoup de prototypes très spécifiques) → coût du modèle élevé mais coût des données bas
    
    Mathématiquement:
    MDL = L(M) + L(D|M) + L(A)
    où:
    - L(M) est la longueur de description du modèle
    - L(D|M) est la longueur de description des données étant donné le modèle
    - L(A) est la longueur de description de l'assignation des instances aux prototypes
    """
    model_cost = calculate_model_cost(clusters)
    data_cost = calculate_data_cost(clusters)
    assign_cost = calculate_assignment_cost(clusters)
    
    total_cost = model_cost + data_cost + assign_cost
    
    # Détails pour analyse et débogage
    cost_details = {
        'model_cost': model_cost,
        'data_cost': data_cost,
        'assignment_cost': assign_cost,
        'total_cost': total_cost
    }
    
    return total_cost, cost_details

# Fonction plus détaillée qui remplace la fonction calculate_total_cost originale
def calculate_total_cost(clusters, verbose=False):
    """
    Version simplifiée de calculate_total_mdl_cost pour compatibilité avec le code existant.
    
    Args:
        clusters: Liste des objets Cluster
        verbose: Si True, affiche les détails du calcul de coût
        
    Returns:
        Le coût MDL total
    """
    total_cost, details = calculate_total_mdl_cost(clusters)
    
    if verbose:
        print(f"Coût du modèle: {details['model_cost']:.2f} bits")
        print(f"Coût des données: {details['data_cost']:.2f} bits")
        print(f"Coût d'assignation: {details['assignment_cost']:.2f} bits")
        print(f"Coût total: {details['total_cost']:.2f} bits")
    
    return total_cost

# Exemple d'utilisation
if __name__ == "__main__":
    # Créer quelques clusters pour démonstration
    c1 = Cluster("0110", 8)  # Prototype: 0110****
    c1.size = 10
    
    c2 = Cluster("101", 8)   # Prototype: 101*****
    c2.size = 5
    
    clusters = [c1, c2]
    
    # Calculer et afficher le coût détaillé
    total_cost, details = calculate_total_mdl_cost(clusters)
    
    print("=== Analyse du coût MDL ===")
    print(f"Cluster 1: {c1.pattern_str()} avec {c1.size} instances")
    print(f"Cluster 2: {c2.pattern_str()} avec {c2.size} instances")
    print("\nDécomposition du coût:")
    print(f"- Coût du modèle: {details['model_cost']} bits")
    print(f"  (somme des longueurs de préfixe: {c1.prefix_length()} + {c2.prefix_length()})")
    print(f"- Coût des données: {details['data_cost']} bits")
    print(f"  (c1: {c1.size}*{c1.wildcards_count()} + c2: {c2.size}*{c2.wildcards_count()})")
    print(f"- Coût d'assignation: {details['assignment_cost']:.2f} bits")
    print(f"  (basé sur la distribution: {c1.size}/{c1.size+c2.size} et {c2.size}/{c1.size+c2.size})")
    print(f"\nCoût total MDL: {details['total_cost']:.2f} bits")
