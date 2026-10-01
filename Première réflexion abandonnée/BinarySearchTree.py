"""
Telecom Paris : projet Contraste

Projet réalisé par : Samy Bali, Hussein El gouch, Issa Daou, Anas El Barhdadi

Encadré par Jean-Louis Dessalles

Ce fichier implémente un arbre binaire de recherche pour trouver efficacement
le point le plus proche dans un espace binaire.
"""

class BinarySearchTree:
    """
    Arbre binaire de recherche pour trouver efficacement les points les plus proches
    dans un espace binaire.
    
    Attributs:
        - value: la valeur (nombre) stockée dans ce nœud
        - bin_rep: représentation binaire du nombre (chaîne de '0' et '1')
        - left: sous-arbre gauche (bits à '1')
        - right: sous-arbre droit (bits à '0')
        - depth: profondeur du nœud dans l'arbre (racine = 0)
        - bit_index: index du bit utilisé pour la décision à ce niveau
    """
    
    def __init__(self, value, bin_rep=None, depth=0, bin_length=20):
        """
        Initialise un nœud de l'arbre binaire de recherche.
        
        Args:
            value: valeur numérique stockée dans ce nœud
            bin_rep: représentation binaire (si None, calculée automatiquement)
            depth: profondeur du nœud dans l'arbre
            bin_length: longueur de la représentation binaire
        """
        self.value = value
        
        # Si la représentation binaire n'est pas fournie, la calculer
        if bin_rep is None:
            self.bin_rep = format(value, f'0{bin_length}b')
        else:
            self.bin_rep = bin_rep
            
        self.left = None     # Sous-arbre gauche (pour les '1')
        self.right = None    # Sous-arbre droit (pour les '0')
        self.depth = depth   # Profondeur dans l'arbre
        self.bit_index = depth % len(self.bin_rep)  # Bit utilisé pour la décision à ce niveau
    
    def insert(self, value, bin_rep=None, bin_length=20):
        """
        Insère une nouvelle valeur dans l'arbre.
        
        Args:
            value: valeur à insérer
            bin_rep: représentation binaire (optionnelle)
            bin_length: longueur de la représentation binaire
        """
        # Calculer la représentation binaire si non fournie
        if bin_rep is None:
            bin_rep = format(value, f'0{bin_length}b')
        
        # Déterminer le bit à utiliser pour la décision
        bit_index = self.depth % len(bin_rep)
        bit_value = bin_rep[bit_index]
        
        # Insérer à gauche si le bit est '1', à droite sinon
        if bit_value == '1':
            if self.left is None:
                self.left = BinarySearchTree(value, bin_rep, self.depth + 1, bin_length)
            else:
                self.left.insert(value, bin_rep, bin_length)
        else:
            if self.right is None:
                self.right = BinarySearchTree(value, bin_rep, self.depth + 1, bin_length)
            else:
                self.right.insert(value, bin_rep, bin_length)
    
    def hamming_distance(self, bin_rep1, bin_rep2):
        """
        Calcule la distance de Hamming entre deux représentations binaires.
        
        Args:
            bin_rep1: première représentation binaire
            bin_rep2: deuxième représentation binaire
            
        Returns:
            Distance de Hamming (nombre de bits différents)
        """
        return sum(b1 != b2 for b1, b2 in zip(bin_rep1, bin_rep2))
    
    def find_nearest(self, target, bin_rep=None, bin_length=20, best_match=None, best_distance=float('inf')):
        """
        Trouve la valeur la plus proche de la cible dans l'arbre.
        
        Args:
            target: valeur cible
            bin_rep: représentation binaire de la cible (optionnelle)
            bin_length: longueur de la représentation binaire
            best_match: meilleure correspondance trouvée jusqu'à présent
            best_distance: meilleure distance trouvée jusqu'à présent
            
        Returns:
            Tuple (valeur la plus proche, distance minimale)
        """
        # Calculer la représentation binaire si non fournie
        if bin_rep is None:
            bin_rep = format(target, f'0{bin_length}b')
        
        # Comparer avec le nœud actuel
        current_distance = self.hamming_distance(self.bin_rep, bin_rep)
        
        if current_distance < best_distance:
            best_match = self.value
            best_distance = current_distance
        
        # Déterminer le bit à utiliser pour la décision
        bit_index = self.depth % len(bin_rep)
        bit_value = bin_rep[bit_index]
        
        # Chercher d'abord dans la direction appropriée
        primary, secondary = None, None
        if bit_value == '1':
            primary, secondary = self.left, self.right
        else:
            primary, secondary = self.right, self.left
        
        # Rechercher dans la direction primaire (correspondant au bit)
        if primary is not None:
            best_match, best_distance = primary.find_nearest(
                target, bin_rep, bin_length, best_match, best_distance
            )
        
        # Si l'autre branche pourrait contenir un point plus proche, y chercher aussi
        # Cela se produit quand la distance à la frontière est inférieure à la meilleure distance
        if secondary is not None:
            # Dans un arbre de recherche basé sur des bits, la "frontière" est à distance 1
            # car nous changeons exactement 1 bit pour passer à l'autre branche
            if 1 < best_distance:  # Si franchir la frontière peut mener à un meilleur résultat
                best_match, best_distance = secondary.find_nearest(
                    target, bin_rep, bin_length, best_match, best_distance
                )
        
        return best_match, best_distance
