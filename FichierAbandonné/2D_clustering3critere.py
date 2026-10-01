# Clustering 2D modifié pour valeurs entières avec options Gray/Binaire
import matplotlib.pyplot as plt
import math
import time
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field
from collections import defaultdict
import random

# ============== CONFIGURATION ==============

@dataclass
class Config:
    """Configuration globale pour l'algorithme de clustering"""
    nb_bits: int = 8  # Nombre de bits pour la représentation
    threshold: int = 1  # Seuil de différence en bits pour rattacher à un cluster
    decay_factor: float = 0.9  # Facteur de décroissance pour la récence
    use_gray_code: bool = True  # True pour Gray code, False pour binaire standard
    

# ============== REPRÉSENTATION BINAIRE ==============

class BinaryRepresentation:
    """Classe pour gérer la conversion et manipulation des représentations binaires"""
    
    @staticmethod
    def int_to_binary(value: int, nb_bits: int) -> str:
        """Convertit un entier en représentation binaire standard"""
        if value < 0:
            raise ValueError("Cette fonction ne traite que les entiers positifs")
        
        max_value = (2 ** nb_bits) - 1
        if value > max_value:
            # Si la valeur dépasse la capacité, on la tronque
            value = max_value
        
        return format(value, f'0{nb_bits}b')
    
    @staticmethod
    def int_to_gray(value: int, nb_bits: int) -> str:
        """Convertit un entier en code de Gray"""
        if value < 0:
            raise ValueError("Cette fonction ne traite que les entiers positifs")
        
        max_value = (2 ** nb_bits) - 1
        if value > max_value:
            value = max_value
        
        # Conversion en code de Gray : G = B XOR (B >> 1)
        gray = value ^ (value >> 1)
        return format(gray, f'0{nb_bits}b')
    
    @staticmethod
    def hamming_distance(bin1: str, bin2: str) -> int:
        """Calcule la distance de Hamming entre deux chaînes binaires"""
        return sum(c1 != c2 for c1, c2 in zip(bin1, bin2))
    
    @staticmethod
    def first_different_bit(bin1: str, bin2: str) -> int:
        """Trouve l'index du premier bit différent entre deux chaînes"""
        for i, (c1, c2) in enumerate(zip(bin1, bin2)):
            if c1 != c2:
                return i
        return -1  # Aucune différence trouvée
    
    @staticmethod
    def validate_integer_input(x, y) -> Tuple[int, int]:
        """Valide et convertit les entrées en entiers positifs"""
        try:
            x_int = int(round(x))
            y_int = int(round(y))
            
            # Assure que les valeurs sont positives
            x_int = max(0, x_int)
            y_int = max(0, y_int)
            
            return x_int, y_int
        except (ValueError, TypeError) as e:
            raise ValueError(f"Les valeurs d'entrée doivent être convertibles en entiers: {e}")


# ============== CLASSES DE DONNÉES ==============

@dataclass
class DataPoint:
    """Représente un point de données en 2D avec des entiers positifs"""
    x: int
    y: int
    binary_x: str = ""
    binary_y: str = ""
    timestamp: float = field(default_factory=time.time)
    cluster_id: Optional[int] = None
    config: Config = field(default_factory=Config)
    
    def __post_init__(self):
        """Initialise les représentations binaires après création"""
        if not self.binary_x or not self.binary_y:
            self._update_binary_representations()
    
    def _update_binary_representations(self):
        """Met à jour les représentations binaires selon la configuration"""
        if self.config.use_gray_code:
            self.binary_x = BinaryRepresentation.int_to_gray(self.x, self.config.nb_bits)
            self.binary_y = BinaryRepresentation.int_to_gray(self.y, self.config.nb_bits)
        else:
            self.binary_x = BinaryRepresentation.int_to_binary(self.x, self.config.nb_bits)
            self.binary_y = BinaryRepresentation.int_to_binary(self.y, self.config.nb_bits)


@dataclass
class Prototype:
    """Représente un prototype avec support pour Gray et binaire standard"""
    binary_x: str  # Peut contenir des '*' pour les bits imprécis
    binary_y: str
    children: List['TreeNode'] = field(default_factory=list)
    instances: Set[int] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    config: Config = field(default_factory=Config)
    
    def update_last_seen(self):
        """Met à jour le timestamp de dernière observation"""
        self.last_seen = time.time()
    
    def matches(self, binary_x: str, binary_y: str) -> bool:
        """Vérifie si un point correspond au prototype"""
        return all(p == '*' or p == b for p, b in zip(self.binary_x, binary_x)) and \
               all(p == '*' or p == b for p, b in zip(self.binary_y, binary_y))
    
    def distance_to(self, binary_x: str, binary_y: str, dimension: str = 'both') -> int:
        """Calcule la distance en bits entre le prototype et un point"""
        if dimension == 'x' or dimension == 'both':
            dist_x = sum(1 for p, b in zip(self.binary_x, binary_x) 
                        if p != '*' and p != b)
            return dist_x
        else:
            dist_x = 0
    
            
        if dimension == 'y' or dimension == 'both':
            dist_y = sum(1 for p, b in zip(self.binary_y, binary_y) 
                        if p != '*' and p != b)
            return dist_y
        else:
            dist_y = 0
            
        return dist_x + dist_y
    
    def to_string(self) -> str:
        """Représentation textuelle du prototype"""
        encoding = "Gray" if self.config.use_gray_code else "Binary"
        return f"Prototype ({encoding}): X={self.binary_x}, Y={self.binary_y}"


# ============== CLASSES PRINCIPALES ==============

class TreeNode:
    """Nœud de l'arbre avec support amélioré pour le tracking des différences"""
    
    def __init__(self, data: DataPoint, parent: Optional['TreeNode'] = None):
        self.data = data
        self.parent = parent
        self.left: Optional['TreeNode'] = None
        self.right: Optional['TreeNode'] = None
        self.prototype: Optional[Prototype] = None
        self.depth = 0 if parent is None else parent.depth + 1
        self.popularity = 0
        self.first_diff_bit_x = -1  # Premier bit différent en X
        self.first_diff_bit_y = -1  # Premier bit différent en Y
        
    def add_child(self, child: 'TreeNode', direction: str):
        """Ajoute un enfant et met à jour la popularité"""
        if direction == 'left':
            self.left = child
        else:
            self.right = child
        
        child.parent = self
        
        # Met à jour la popularité des ancêtres
        current = self
        while current:
            current.popularity += 1
            current = current.parent
    
    def find_first_diff_with_parent(self):
        """Trouve le premier bit différent avec le parent"""
        if self.parent:
            self.first_diff_bit_x = BinaryRepresentation.first_different_bit(
                self.data.binary_x, self.parent.data.binary_x
            )
            self.first_diff_bit_y = BinaryRepresentation.first_different_bit(
                self.data.binary_y, self.parent.data.binary_y
            )
    
    def create_prototype(self, bit_position: int = None, dimension: str = 'both') -> Prototype:
        """Crée un prototype en remplaçant certains bits par des '*'"""
        if bit_position is None:
            # Par défaut, utilise la moitié des bits
            bit_position = self.data.config.nb_bits // 2
        
        binary_x = list(self.data.binary_x)
        binary_y = list(self.data.binary_y)
        
        if dimension in ['x', 'both']:
            for i in range(bit_position, len(binary_x)):
                binary_x[i] = '*'
                
        if dimension in ['y', 'both']:
            for i in range(bit_position, len(binary_y)):
                binary_y[i] = '*'
        
        prototype = Prototype(''.join(binary_x), ''.join(binary_y), config=self.data.config)
        return prototype


class ScoringSystem:
    """Système de scoring adapté aux entiers et aux deux types d'encodage"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def tree_score(self, node: TreeNode) -> float:
        """Score basé sur la position dans l'arbre (MDL)"""
        # En binaire standard, privilégier les différences sur les bits de poids fort
        if not self.config.use_gray_code:
            # Bonus si la différence est sur les premiers bits (poids fort)
            weight = 1.0
            if node.first_diff_bit_x >= 0:
                weight += (self.config.nb_bits - node.first_diff_bit_x) / self.config.nb_bits
            if node.first_diff_bit_y >= 0:
                weight += (self.config.nb_bits - node.first_diff_bit_y) / self.config.nb_bits
            weight /= 3  # Normaliser
            
            return weight / (1.0 + math.log(1 + node.depth))
        else:
            # En Gray, tous les bits ont le même poids
            return 1.0 / (1.0 + math.log(1 + node.depth))
    
    def popularity_score(self, node: TreeNode) -> float:
        """Score basé sur la popularité"""
        return 1.0 - 1.0 / (1.0 + node.popularity / 10.0)
    
    def recency_score(self, prototype: Prototype, current_time: float) -> float:
        """Score basé sur la récence"""
        time_diff = current_time - prototype.last_seen
        return math.exp(-time_diff * (1 - self.config.decay_factor))
    
    def combined_score(self, node: TreeNode, current_time: float) -> float:
        """Score combiné avec prise en compte du type d'encodage"""
        tree = self.tree_score(node)
        popularity = self.popularity_score(node)
        
        if node.prototype:
            recency = self.recency_score(node.prototype, current_time)
        else:
            recency = 0.5
            
        return min(tree, popularity, recency)


class IncrementalClustering:
    """Algorithme de clustering adapté aux entiers avec options Gray/Binaire"""
    
    def __init__(self, config: Config = Config()):
        self.config = config
        self.data_points: List[DataPoint] = []
        self.tree_root: Optional[TreeNode] = None
        self.prototypes: List[Prototype] = []
        self.scoring = ScoringSystem(config)
        self.next_cluster_id = 0
        
    def add_data_point(self, x, y) -> Tuple[int, bool]:
        """Ajoute un point entier positif et le clusterise"""
        # Validation et conversion en entiers
        x_int, y_int = BinaryRepresentation.validate_integer_input(x, y)
        
        # Création du point avec la configuration
        point = DataPoint(x_int, y_int, config=self.config)
        self.data_points.append(point)
        
        if self.tree_root is None:
            self.tree_root = TreeNode(point)
            point.cluster_id = self.next_cluster_id
            self.next_cluster_id += 1
            return point.cluster_id, True
        
        best_prototype, best_score = self._find_best_prototype(point)
        
        if best_prototype and self._should_attach_to_prototype(point, best_prototype):
            # Réutilise un cluster ID existant ou crée un nouveau
            point.cluster_id = self._get_cluster_id_for_prototype(best_prototype)
            best_prototype.instances.add(len(self.data_points) - 1)
            best_prototype.update_last_seen()
            return point.cluster_id, False
        else:
            self._create_new_cluster(point)
            return point.cluster_id, True
    
    def _get_cluster_id_for_prototype(self, prototype: Prototype) -> int:
        """Récupère ou crée un cluster ID pour un prototype"""
        # Cherche dans les points existants couverts par ce prototype
        for idx in prototype.instances:
            if idx < len(self.data_points):
                return self.data_points[idx].cluster_id
        
        # Si aucun cluster ID trouvé, crée un nouveau
        new_id = self.next_cluster_id
        self.next_cluster_id += 1
        return new_id
    
    def _find_best_prototype(self, point: DataPoint) -> Tuple[Optional[Prototype], float]:
        """Trouve le meilleur prototype pour un point donné"""
        best_prototype = None
        best_score = float('inf')
        best_distance = float('inf')
        
        for prototype in self.prototypes:
            # Calcul des distances par dimension
            dist_x = prototype.distance_to(point.binary_x, point.binary_y, 'x')
            dist_y = prototype.distance_to(point.binary_x, point.binary_y, 'y')
            
            # Gestion des deux problèmes spécifiques au 2D
            
            # Problème 1: Proximité mixte
            if (dist_x <= self.config.threshold) != (dist_y <= self.config.threshold):
                total_distance = dist_x + dist_y
                
                # Vérification du seuil pour la dimension éloignée
                if max(dist_x, dist_y) <= self.config.threshold:
                    current_score = self._calculate_prototype_score(prototype, point)
                    if total_distance < best_distance or \
                       (total_distance == best_distance and current_score < best_score):
                        best_prototype = prototype
                        best_score = current_score
                        best_distance = total_distance
            
            # Problème 2: Proximité multiple
            elif dist_x <= self.config.threshold and dist_y <= self.config.threshold:
                total_distance = dist_x + dist_y
                current_score = self._calculate_prototype_score(prototype, point)
                
                # Choix du cluster avec la distance minimale
                if total_distance < best_distance or \
                   (total_distance == best_distance and current_score < best_score):
                    best_prototype = prototype
                    best_score = current_score
                    best_distance = total_distance
        
        return best_prototype, best_score
    
    def _should_attach_to_prototype(self, point: DataPoint, prototype: Prototype) -> bool:
        """Détermine si un point doit être rattaché à un prototype"""
        distx = prototype.distance_to(point.binary_x, point.binary_y, 'x')
        disty = prototype.distance_to(point.binary_x, point.binary_y, 'y')
        if distx <= self.config.threshold and disty <= self.config.threshold:
            return True
        return False
    
    def _calculate_prototype_score(self, prototype: Prototype, point: DataPoint) -> float:
        """Calcule le score combiné pour un prototype donné"""
        node = self._find_node_for_prototype(prototype)
        if node:
            return self.scoring.combined_score(node, time.time())
        return float('inf')
    
    def _find_node_for_prototype(self, prototype: Prototype) -> Optional[TreeNode]:
        """Trouve le nœud de l'arbre correspondant à un prototype"""
        queue = [self.tree_root]
        while queue:
            node = queue.pop(0)
            if node.prototype == prototype:
                return node
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        return None
    
    def _create_new_cluster(self, point: DataPoint):
        """Crée un nouveau cluster pour un point"""
        new_node = TreeNode(point)
        point.cluster_id = self.next_cluster_id
        self.next_cluster_id += 1
        
        # Insertion dans l'arbre
        self._insert_into_tree(new_node)
        
        # Création du prototype
        prototype = new_node.create_prototype(self.config.nb_bits // 2, 'both')
        new_node.prototype = prototype
        prototype.instances.add(len(self.data_points) - 1)
        self.prototypes.append(prototype)
    
    def _insert_into_tree(self, new_node: TreeNode):
        """Insère un nouveau nœud dans l'arbre selon l'encodage"""
        current = self.tree_root
        
        while True:
            # Calcul des distances de Hamming
            dist_x = BinaryRepresentation.hamming_distance(
                current.data.binary_x, 
                new_node.data.binary_x
            )
            dist_y = BinaryRepresentation.hamming_distance(
                current.data.binary_y, 
                new_node.data.binary_y
            )
            
            if dist_x + dist_y == 0:
                break  # Points identiques
            
            # Trouve le premier bit différent pour déterminer la direction
            new_node.find_first_diff_with_parent()
            
            # Stratégie d'insertion selon l'encodage
            if self.config.use_gray_code:
                # En Gray code, utilise le premier bit différent
                first_diff_x = BinaryRepresentation.first_different_bit(
                    current.data.binary_x, new_node.data.binary_x
                )
                first_diff_y = BinaryRepresentation.first_different_bit(
                    current.data.binary_y, new_node.data.binary_y
                )
                
                # Privilégie la dimension avec la première différence
                if first_diff_x >= 0 and (first_diff_y < 0 or first_diff_x <= first_diff_y):
                    use_bit = new_node.data.binary_x[first_diff_x] == '1'
                else:
                    use_bit = new_node.data.binary_y[first_diff_y] == '1'
            else:
                # En binaire standard, privilégie les bits de poids fort
                # Cherche la première différence significative
                use_bit = False
                for i in range(self.config.nb_bits):
                    if i < len(current.data.binary_x) and current.data.binary_x[i] != new_node.data.binary_x[i]:
                        use_bit = new_node.data.binary_x[i] == '1'
                        break
                    elif i < len(current.data.binary_y) and current.data.binary_y[i] != new_node.data.binary_y[i]:
                        use_bit = new_node.data.binary_y[i] == '1'
                        break
            
            # Insertion selon la direction déterminée
            if not use_bit:  # Aller à gauche
                if current.left is None:
                    current.left = new_node
                    current.add_child(new_node, 'left')
                    break
                else:
                    current = current.left
            else:  # Aller à droite
                if current.right is None:
                    current.right = new_node
                    current.add_child(new_node, 'right')
                    break
                else:
                    current = current.right


# ============== VISUALISATION ADAPTÉE ==============

class IntegerClusteringVisualizer:
    """Visualisation adaptée aux entiers avec comparaison Gray/Binaire"""
    
    def __init__(self, clustering: IncrementalClustering):
        self.clustering = clustering
    
    def plot_2d_scatter(self, figsize=(10, 8), show_binary=True, save_path=None):
        """Scatter plot avec informations binaires"""
        plt.figure(figsize=figsize)
        
        # Extraction des données
        x_coords = [p.x for p in self.clustering.data_points]
        y_coords = [p.y for p in self.clustering.data_points]
        clusters = [p.cluster_id for p in self.clustering.data_points]
        
        # Scatter plot
        scatter = plt.scatter(x_coords, y_coords, c=clusters, 
                            cmap='rainbow', s=100, alpha=0.7, edgecolors='black')
        
        # Annotations avec valeurs binaires si demandé
        if show_binary:
            for i, point in enumerate(self.clustering.data_points):
                encoding = "G" if self.clustering.config.use_gray_code else "B"
                annotation = f"({point.x},{point.y})\n{encoding}X:{point.binary_x}\n{encoding}Y:{point.binary_y}"
                plt.annotate(annotation, (point.x, point.y), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
        
        # Centres des prototypes
        for i, prototype in enumerate(self.clustering.prototypes):
            # Trouve les points associés
            associated_points = []
            for idx in prototype.instances:
                if idx < len(self.clustering.data_points):
                    associated_points.append(self.clustering.data_points[idx])
            
            if associated_points:
                center_x = sum(p.x for p in associated_points) / len(associated_points)
                center_y = sum(p.y for p in associated_points) / len(associated_points)
                plt.scatter(center_x, center_y, marker='*', s=300, 
                          c='black', edgecolors='white', linewidth=2)
        
        encoding_name = "Code de Gray" if self.clustering.config.use_gray_code else "Binaire Standard"
        plt.title(f"Clustering 2D - Entiers avec {encoding_name}")
        plt.xlabel('X (entier)')
        plt.ylabel('Y (entier)')
        plt.colorbar(scatter, label='Cluster ID')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_encodings(self, test_data: List[Tuple[int, int]], figsize=(15, 6)):
        """Compare les résultats Gray vs Binaire sur les mêmes données"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Test avec Gray
        config_gray = Config(
            nb_bits=self.clustering.config.nb_bits,
            threshold=self.clustering.config.threshold,
            use_gray_code=True
        )
        clustering_gray = IncrementalClustering(config_gray)
        
        for x, y in test_data:
            clustering_gray.add_data_point(x, y)
        
        # Test avec Binaire
        config_binary = Config(
            nb_bits=self.clustering.config.nb_bits,
            threshold=self.clustering.config.threshold,
            use_gray_code=False
        )
        clustering_binary = IncrementalClustering(config_binary)
        
        for x, y in test_data:
            clustering_binary.add_data_point(x, y)
        
        # Visualisation Gray
        x_coords = [p.x for p in clustering_gray.data_points]
        y_coords = [p.y for p in clustering_gray.data_points]
        clusters = [p.cluster_id for p in clustering_gray.data_points]
        
        ax1.scatter(x_coords, y_coords, c=clusters, cmap='rainbow', s=100)
        ax1.set_title(f"Code de Gray\n{clustering_gray.next_cluster_id} clusters")
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.grid(True, alpha=0.3)
        
        # Visualisation Binaire
        x_coords = [p.x for p in clustering_binary.data_points]
        y_coords = [p.y for p in clustering_binary.data_points]
        clusters = [p.cluster_id for p in clustering_binary.data_points]
        
        ax2.scatter(x_coords, y_coords, c=clusters, cmap='rainbow', s=100)
        ax2.set_title(f"Binaire Standard\n{clustering_binary.next_cluster_id} clusters")
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return clustering_gray, clustering_binary


# ============== DÉMONSTRATION ==============

def demonstrate_integer_clustering():
    """Démonstration complète avec entiers et comparaison encodages"""
    
    print("=== Clustering 2D pour Entiers - Gray vs Binaire ===\n")
    
    # Configuration de test
    config = Config(nb_bits=6, threshold=2, use_gray_code=True)
    
    # Données de test (entiers)
    test_points = [
        (1, 1), (2, 2), (3, 1),    # Cluster 1
        (15, 15), (14, 16), (16, 14),  # Cluster 2
        (8, 2), (7, 3), (9, 1),    # Cluster 3
        (1, 15), (2, 14),          # Points isolés
    ]
    
    print("Points de test:")
    for i, (x, y) in enumerate(test_points):
        binary_x = BinaryRepresentation.int_to_binary(x, config.nb_bits)
        binary_y = BinaryRepresentation.int_to_binary(y, config.nb_bits)
        gray_x = BinaryRepresentation.int_to_gray(x, config.nb_bits)
        gray_y = BinaryRepresentation.int_to_gray(y, config.nb_bits)
        
        print(f"  Point {i}: ({x}, {y})")
        print(f"    Binaire: X={binary_x}, Y={binary_y}")
        print(f"    Gray:    X={gray_x}, Y={gray_y}")
    
    # Test avec Code de Gray
    print(f"\n=== Test avec Code de Gray ===")
    clustering_gray = IncrementalClustering(Config(nb_bits=6, threshold=2, use_gray_code=True))
    
    for x, y in test_points:
        cluster_id, is_new = clustering_gray.add_data_point(x, y)
        print(f"Point ({x}, {y}) -> Cluster {cluster_id} (nouveau: {is_new})")
    
    # Test avec Binaire Standard
    print(f"\n=== Test avec Binaire Standard ===")
    clustering_binary = IncrementalClustering(Config(nb_bits=6, threshold=2, use_gray_code=False))
    
    for x, y in test_points:
        cluster_id, is_new = clustering_binary.add_data_point(x, y)
        print(f"Point ({x}, {y}) -> Cluster {cluster_id} (nouveau: {is_new})")
    
    # Comparaison des résultats
    print(f"\n=== Comparaison des résultats ===")
    print(f"Code de Gray:     {clustering_gray.next_cluster_id} clusters")
    print(f"Binaire Standard: {clustering_binary.next_cluster_id} clusters")
    
    # Visualisation
    visualizer_gray = IntegerClusteringVisualizer(clustering_gray)
    print(f"\nVisualisation Code de Gray:")
    visualizer_gray.plot_2d_scatter(show_binary=True)
    
    visualizer_binary = IntegerClusteringVisualizer(clustering_binary)
    print(f"\nVisualisation Binaire Standard:")
    visualizer_binary.plot_2d_scatter(show_binary=True)
    
    # Comparaison côte à côte
    print(f"\nComparaison côte à côte:")
    visualizer_gray.compare_encodings(test_points)
    
    return clustering_gray, clustering_binary


def example_differences():
    """Exemple illustrant les différences entre Gray et Binaire"""
    print("\n=== Exemple des différences d'encodage ===")
    
    # Séquence de nombres consécutifs
    numbers = list(range(8))
    
    print("Différences entre nombres consécutifs :")
    print("Nombre | Binaire  | Gray     | Dist.Ham Bin | Dist.Ham Gray")
    print("-------|----------|----------|--------------|---------------")
    
    for i in range(len(numbers) - 1):
        curr = numbers[i]
        next_num = numbers[i + 1]
        
        bin_curr = BinaryRepresentation.int_to_binary(curr, 4)
        bin_next = BinaryRepresentation.int_to_binary(next_num, 4)
        gray_curr = BinaryRepresentation.int_to_gray(curr, 4)
        gray_next = BinaryRepresentation.int_to_gray(next_num, 4)
        
        ham_bin = BinaryRepresentation.hamming_distance(bin_curr, bin_next)
        ham_gray = BinaryRepresentation.hamming_distance(gray_curr, gray_next)
        
        print(f"{curr:>3} -> {next_num:>1} | {bin_curr} | {gray_curr} | {ham_bin:>11} | {ham_gray:>12}")
    
    print("\nObservation: En code de Gray, les nombres consécutifs ne diffèrent que d'un bit!")


def analyze_clustering_differences():
    """Analyse des différences de clustering entre Gray et Binaire"""
    print("\n=== Analyse des différences de clustering ===")
    
    # Points problématiques pour illustrer la différence
    problematic_points = [
        (7, 5),    # 0111, 0101 en binaire
        (8, 6),    # 1000, 0110 en binaire (grande différence de Hamming)
        (15, 7),   # 1111, 0111 en binaire
        (16, 8),   # Sortie de la plage 4 bits
    ]
    
    print("Analyse de points problématiques:")
    
    for x, y in problematic_points:
        print(f"\nPoint ({x}, {y}):")
        
        # Binaire 4 bits
        try:
            bin_x = BinaryRepresentation.int_to_binary(x, 4)
            bin_y = BinaryRepresentation.int_to_binary(y, 4)
            print(f"  Binaire: X={bin_x}, Y={bin_y}")
        except:
            print(f"  Binaire: Hors limites pour 4 bits")
        
        # Gray 4 bits  
        try:
            gray_x = BinaryRepresentation.int_to_gray(x, 4)
            gray_y = BinaryRepresentation.int_to_gray(y, 4)
            print(f"  Gray:    X={gray_x}, Y={gray_y}")
        except:
            print(f"  Gray: Hors limites pour 4 bits")


def test_specific_differences():
    """Test des cas spécifiques mentionnés (ex: 1011 vs 1001)"""
    print("\n=== Test des différences spécifiques ===")
    
    # Exemple donné: 1011 diffère de 1001 au 3ème bit
    print("Exemple: 1011 vs 1001")
    bin1, bin2 = "1011", "1001"
    
    first_diff = BinaryRepresentation.first_different_bit(bin1, bin2)
    ham_dist = BinaryRepresentation.hamming_distance(bin1, bin2)
    
    print(f"  Première différence: bit {first_diff} (0-indexé)")
    print(f"  Distance de Hamming: {ham_dist}")
    
    # Conversion vers entiers
    int1 = int(bin1, 2)  # 11
    int2 = int(bin2, 2)  # 9
    
    print(f"  Entiers correspondants: {int1} et {int2}")
    
    # Test dans le clustering
    config = Config(nb_bits=4, threshold=1, use_gray_code=False)
    clustering = IncrementalClustering(config)
    
    # Ajouter quelques points pour créer un contexte
    clustering.add_data_point(int1, int1)  # (11, 11)
    clustering.add_data_point(int2, int2)  # (9, 9)
    
    print(f"\nRésultat clustering:")
    print(f"  Point ({int1}, {int1}) -> Cluster {clustering.data_points[0].cluster_id}")
    print(f"  Point ({int2}, {int2}) -> Cluster {clustering.data_points[1].cluster_id}")
    
    if clustering.data_points[0].cluster_id == clustering.data_points[1].cluster_id:
        print("  -> Même cluster (distance acceptable)")
    else:
        print("  -> Clusters différents (distance trop grande)")


def main():
    """Fonction principale pour tester toutes les fonctionnalités"""
    
    # Démonstration complète
    demonstrate_integer_clustering()
    
    # Exemples des différences d'encodage
    example_differences()
    
    # Analyse des différences de clustering
    analyze_clustering_differences()
    
    # Test des cas spécifiques
    test_specific_differences()
    
    # Test interactif optionnel
    print("\n=== Test Interactif ===")
    response = input("Voulez-vous tester vos propres valeurs? (o/n): ")
    
    if response.lower() == 'o':
        interactive_test()


def interactive_test():
    """Test interactif pour entrer ses propres valeurs"""
    print("\nTest interactif - Entrez vos propres valeurs entières positives")
    
    try:
        nb_bits = int(input("Nombre de bits (4-16): "))
        threshold = int(input("Seuil de clustering (1-8): "))
        
        use_gray = input("Utiliser le code de Gray? (o/n): ").lower() == 'o'
        
        config = Config(nb_bits=nb_bits, threshold=threshold, use_gray_code=use_gray)
        clustering = IncrementalClustering(config)
        
        print(f"\nConfiguration: {nb_bits} bits, seuil={threshold}, Gray={use_gray}")
        print("Entrez des points (x,y). Tapez 'q' pour quitter.")
        
        while True:
            try:
                user_input = input("\nPoint (x,y) ou 'q': ")
                if user_input.lower() == 'q':
                    break
                
                # Parse input
                parts = user_input.replace('(', '').replace(')', '').split(',')
                x = int(parts[0].strip())
                y = int(parts[1].strip())
                
                cluster_id, is_new = clustering.add_data_point(x, y)
                point = clustering.data_points[-1]
                
                print(f"  Point ({x}, {y}):")
                print(f"  -> Binaire: X={point.binary_x}, Y={point.binary_y}")
                print(f"  -> Cluster {cluster_id} {'(nouveau)' if is_new else '(existant)'}")
                
            except (ValueError, IndexError):
                print("Format invalide. Utilisez: x,y (ex: 5,3)")
            except Exception as e:
                print(f"Erreur: {e}")
        
        # Visualisation finale
        if len(clustering.data_points) > 0:
            print(f"\nRésultats finaux: {clustering.next_cluster_id} clusters créés")
            visualizer = IntegerClusteringVisualizer(clustering)
            visualizer.plot_2d_scatter(show_binary=True)
    
    except Exception as e:
        print(f"Erreur lors du test interactif: {e}")


# ============== UTILITAIRES SUPPLÉMENTAIRES ==============

def create_test_datasets():
    """Crée différents jeux de données pour tester les algorithmes"""
    datasets = {
        "consecutifs": [(i, i) for i in range(10)],
        "puissances_2": [(2**i, 2**i) for i in range(6)],
        "bordures": [(0, 0), (1, 0), (0, 1), (1, 1)],
        "aleatoire": [(random.randint(0, 31), random.randint(0, 31)) for _ in range(20)]
    }
    
    return datasets


def benchmark_encodings():
    """Compare les performances des deux encodages"""
    print("\n=== Benchmark des encodages ===")
    
    datasets = create_test_datasets()
    results = {}
    
    for name, data in datasets.items():
        print(f"\nDataset: {name}")
        
        # Test Gray
        config_gray = Config(nb_bits=6, threshold=2, use_gray_code=True)
        clustering_gray = IncrementalClustering(config_gray)
        
        for x, y in data:
            clustering_gray.add_data_point(x, y)
        
        # Test Binaire
        config_binary = Config(nb_bits=6, threshold=2, use_gray_code=False)
        clustering_binary = IncrementalClustering(config_binary)
        
        for x, y in data:
            clustering_binary.add_data_point(x, y)
        
        results[name] = {
            'gray_clusters': clustering_gray.next_cluster_id,
            'binary_clusters': clustering_binary.next_cluster_id,
            'points': len(data)
        }
        
        print(f"  Points: {len(data)}")
        print(f"  Gray: {clustering_gray.next_cluster_id} clusters")
        print(f"  Binaire: {clustering_binary.next_cluster_id} clusters")
    
    # Résumé
    print(f"\n=== Résumé du benchmark ===")
    for name, res in results.items():
        ratio = res['binary_clusters'] / res['gray_clusters'] if res['gray_clusters'] > 0 else 0
        print(f"{name:>12}: Gray={res['gray_clusters']:>2}, Binaire={res['binary_clusters']:>2}, Ratio={ratio:.2f}")


if __name__ == "__main__":
    # Test principal
    main()
    
    # Benchmark supplémentaire
    benchmark_encodings()