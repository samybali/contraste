# Imports nécessaires
from typing import List, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

class TreeNode:
    """Nœud de l'arbre binaire contenant un nombre entier positif ou nul"""
    
    def __init__(self, value: int):
        if value < 0:
            raise ValueError("Seuls les nombres positifs ou nuls sont autorisés")
        
        self.value = value
        self.left = None   # Enfant gauche (bit de différence = 0)
        self.right = None  # Enfant droit (bit de différence = 1)
        self.parent = None # Référence vers le parent
    
    def __str__(self):
        return f"Node({self.value})"
    
    def __repr__(self):
        return self.__str__()


class BinaryTreeBitLogic:
    """Arbre binaire utilisant la logique de bits pour l'organisation"""
    
    def __init__(self, nb_bits: int = 32):
        """
        Initialise l'arbre vide (la racine sera la première donnée ajoutée)
        
        Args:
            nb_bits: Nombre de bits à considérer pour les comparaisons
        """
        self.root = None  # L'arbre commence vide
        self.nb_bits = nb_bits
        self.size = 0
    
    def _find_first_different_bit(self, value1: int, value2: int) -> int:
        """
        Trouve l'index du premier bit différent entre deux valeurs
        
        Args:
            value1, value2: Les deux valeurs à comparer
            
        Returns:
            Index du premier bit différent (0 = bit le plus à droite)
            -1 si les valeurs sont identiques
        """
        xor_result = value1 ^ value2
        
        if xor_result == 0:
            return -1
        
        # Trouve la position du bit le plus significatif différent
        bit_position = 0
        while xor_result > 0:
            if xor_result & 1:
                break
            xor_result >>= 1
            bit_position += 1
        
        return bit_position
    
    def _get_bit_value(self, value: int, bit_position: int) -> int:
        """
        Récupère la valeur d'un bit à une position donnée
        
        Args:
            value: La valeur à examiner
            bit_position: Position du bit (0 = bit le plus à droite)
            
        Returns:
            0 ou 1 selon la valeur du bit
        """
        return (value >> bit_position) & 1
    
    def insert(self, value: int) -> bool:
        """
        Insère une valeur dans l'arbre
        
        Args:
            value: Valeur entière positive ou nulle à insérer
            
        Returns:
            True si l'insertion a réussi, False si la valeur existe déjà
        """
        if value < 0:
            raise ValueError("Seuls les nombres positifs ou nuls sont autorisés")
        
        # Cas spécial : premier élément (racine)
        if self.root is None:
            self.root = TreeNode(value)
            self.size = 1
            print(f"Première insertion : {value} devient la racine")
            return True
        
        return self._insert_recursive(self.root, value, None)
    
    def _insert_recursive(self, current_node: TreeNode, value: int, parent: TreeNode) -> bool:
        """
        Insertion récursive selon la logique de bits
        
        Args:
            current_node: Nœud actuel dans la traversée
            value: Valeur à insérer
            parent: Nœud parent du nœud actuel
            
        Returns:
            True si l'insertion a réussi
        """
        # Cas 1: La valeur existe déjà
        if current_node.value == value:
            return False
        
        # Trouve le premier bit différent
        bit_diff_pos = self._find_first_different_bit(current_node.value, value)
        
        if bit_diff_pos == -1:  # Valeurs identiques (ne devrait pas arriver)
            return False
        
        # Détermine la direction selon le bit de différence
        bit_value = self._get_bit_value(value, bit_diff_pos)
        
        if bit_value == 1:  # Aller à droite
            if current_node.right is None:
                # Cas simple: insertion directe
                new_node = TreeNode(value)
                current_node.right = new_node
                new_node.parent = current_node
                self.size += 1
                return True
            else:
                # Vérifier s'il faut insérer entre current_node et son enfant droit
                next_node = current_node.right
                next_bit_diff = self._find_first_different_bit(current_node.value, next_node.value)
                
                if bit_diff_pos > next_bit_diff:
                    # Insérer entre current_node et next_node
                    new_node = TreeNode(value)
                    current_node.right = new_node
                    new_node.parent = current_node
                    new_node.right = next_node
                    next_node.parent = new_node
                    self.size += 1
                    return True
                else:
                    # Continuer la recherche récursivement
                    return self._insert_recursive(next_node, value, current_node)
        
        else:  # bit_value == 0, aller à gauche
            if current_node.left is None:
                # Cas simple: insertion directe
                new_node = TreeNode(value)
                current_node.left = new_node
                new_node.parent = current_node
                self.size += 1
                return True
            else:
                # Vérifier s'il faut insérer entre current_node et son enfant gauche
                next_node = current_node.left
                next_bit_diff = self._find_first_different_bit(current_node.value, next_node.value)
                
                if bit_diff_pos > next_bit_diff:
                    # Insérer entre current_node et next_node
                    new_node = TreeNode(value)
                    current_node.left = new_node
                    new_node.parent = current_node
                    new_node.left = next_node
                    next_node.parent = new_node
                    self.size += 1
                    return True
                else:
                    # Continuer la recherche récursivement
                    return self._insert_recursive(next_node, value, current_node)
    
    def search(self, value: int) -> TreeNode:
        """
        Recherche une valeur dans l'arbre
        
        Args:
            value: Valeur à rechercher
            
        Returns:
            Le nœud contenant la valeur, ou None si non trouvé
        """
        return self._search_recursive(self.root, value)
    
    def _search_recursive(self, current_node: TreeNode, value: int) -> TreeNode:
        """
        Recherche récursive d'une valeur
        
        Args:
            current_node: Nœud actuel
            value: Valeur recherchée
            
        Returns:
            Le nœud contenant la valeur ou None
        """
        if current_node is None:
            return None
        
        if current_node.value == value:
            return current_node
        
        # Trouve le premier bit différent
        bit_diff_pos = self._find_first_different_bit(current_node.value, value)
        
        if bit_diff_pos == -1:  # Valeurs identiques
            return current_node
        
        # Détermine la direction
        bit_value = self._get_bit_value(value, bit_diff_pos)
        
        if bit_value == 1:
            return self._search_recursive(current_node.right, value)
        else:
            return self._search_recursive(current_node.left, value)
    
    def delete(self, value: int) -> bool:
        """
        Supprime une valeur de l'arbre
        
        Args:
            value: Valeur à supprimer
            
        Returns:
            True si la suppression a réussi, False si la valeur n'existe pas
        """
        if self.root is None:
            return False
        
        # Cas spécial : suppression de la racine
        if self.root.value == value:
            return self._delete_root()
        
        node_to_delete = self.search(value)
        if node_to_delete is None:
            return False
        
        return self._delete_recursive(node_to_delete)
    
    def _delete_root(self) -> bool:
        """
        Supprime la racine de l'arbre
        
        Returns:
            True si la suppression a réussi
        """
        if self.root is None:
            return False
        
        # Cas 1: Racine sans enfants
        if self.root.left is None and self.root.right is None:
            self.root = None
            self.size = 0
            return True
        
        # Cas 2: Racine avec un seul enfant
        elif self.root.left is None:
            self.root = self.root.right
            self.root.parent = None
            self.size -= 1
            return True
        
        elif self.root.right is None:
            self.root = self.root.left
            self.root.parent = None
            self.size -= 1
            return True
        
        # Cas 3: Racine avec deux enfants
        # On choisit l'enfant gauche comme nouvelle racine
        # et on rattache l'enfant droit
        else:
            old_right = self.root.right
            self.root = self.root.left
            self.root.parent = None
            
            # Rattache l'ancien sous-arbre droit
            self._attach_subtree(self.root, old_right)
            self.size -= 1
            return True
    
    def _delete_recursive(self, node: TreeNode) -> bool:
        """
        Suppression récursive d'un nœud
        
        Args:
            node: Nœud à supprimer
            
        Returns:
            True si la suppression a réussi
        """
        parent = node.parent
        
        # Cas 1: Nœud feuille (pas d'enfants)
        if node.left is None and node.right is None:
            if parent:
                if parent.left == node:
                    parent.left = None
                else:
                    parent.right = None
            self.size -= 1
            return True
        
        # Cas 2: Nœud avec un seul enfant
        elif node.left is None:
            # Seul enfant droit
            if parent:
                if parent.left == node:
                    parent.left = node.right
                else:
                    parent.right = node.right
                node.right.parent = parent
            else:
                # Le nœud à supprimer est la racine
                self.root = node.right
                node.right.parent = None
            self.size -= 1
            return True
        
        elif node.right is None:
            # Seul enfant gauche
            if parent:
                if parent.left == node:
                    parent.left = node.left
                else:
                    parent.right = node.left
                node.left.parent = parent
            else:
                # Le nœud à supprimer est la racine
                self.root = node.left
                node.left.parent = None
            self.size -= 1
            return True
        
        # Cas 3: Nœud avec deux enfants
        else:
            # Selon votre description: le parent prend la place du nœud supprimé
            # Les sous-arbres deviennent les enfants du parent
            
            if parent is None:
                # Cas spécial: suppression de la racine avec deux enfants
                # On remplace par l'enfant gauche et on rattache l'enfant droit
                self.root = node.left
                node.left.parent = None
                
                # Trouve une place pour rattacher le sous-arbre droit
                self._attach_subtree(self.root, node.right)
            else:
                # Le parent "absorbe" les enfants du nœud supprimé
                left_child = node.left
                right_child = node.right
                
                # Supprime le nœud de son parent
                if parent.left == node:
                    parent.left = None
                else:
                    parent.right = None
                
                # Rattache les sous-arbres au parent
                self._attach_subtree(parent, left_child)
                self._attach_subtree(parent, right_child)
            
            self.size -= 1
            return True
    
    def _attach_subtree(self, parent: TreeNode, subtree_root: TreeNode):
        """
        Rattache un sous-arbre à un parent selon la logique de bits
        
        Args:
            parent: Nœud parent où rattacher
            subtree_root: Racine du sous-arbre à rattacher
        """
        if subtree_root is None:
            return
        
        # Détermine où rattacher selon la logique de bits
        bit_diff_pos = self._find_first_different_bit(parent.value, subtree_root.value)
        
        if bit_diff_pos >= 0:
            bit_value = self._get_bit_value(subtree_root.value, bit_diff_pos)
            
            if bit_value == 1:  # Rattacher à droite
                if parent.right is None:
                    parent.right = subtree_root
                    subtree_root.parent = parent
                else:
                    # Il faut insérer récursivement
                    self._insert_subtree_recursive(parent.right, subtree_root)
            else:  # Rattacher à gauche
                if parent.left is None:
                    parent.left = subtree_root
                    subtree_root.parent = parent
                else:
                    # Il faut insérer récursivement
                    self._insert_subtree_recursive(parent.left, subtree_root)
    
    def _insert_subtree_recursive(self, current: TreeNode, subtree: TreeNode):
        """
        Insère récursivement un sous-arbre complet
        
        Args:
            current: Nœud actuel dans l'arbre principal
            subtree: Racine du sous-arbre à insérer
        """
        if subtree is None:
            return
        
        # Insère d'abord la racine du sous-arbre
        self.insert(subtree.value)
        
        # Puis insère récursivement les enfants
        self._insert_subtree_recursive(current, subtree.left)
        self._insert_subtree_recursive(current, subtree.right)
    
    def inorder_traversal(self) -> List[int]:
        """
        Parcours inordre de l'arbre
        
        Returns:
            Liste des valeurs dans l'ordre inordre
        """
        result = []
        self._inorder_recursive(self.root, result)
        return result
    
    def _inorder_recursive(self, node: TreeNode, result: List[int]):
        """Parcours inordre récursif"""
        if node is not None:
            self._inorder_recursive(node.left, result)
            result.append(node.value)
            self._inorder_recursive(node.right, result)
    
    def preorder_traversal(self) -> List[int]:
        """
        Parcours préordre de l'arbre
        
        Returns:
            Liste des valeurs dans l'ordre préordre
        """
        result = []
        self._preorder_recursive(self.root, result)
        return result
    
    def _preorder_recursive(self, node: TreeNode, result: List[int]):
        """Parcours préordre récursif"""
        if node is not None:
            result.append(node.value)
            self._preorder_recursive(node.left, result)
            self._preorder_recursive(node.right, result)
    
    def display_tree(self, node: TreeNode = None, level: int = 0, prefix: str = "Root: "):
        """
        Affiche l'arbre de manière visuelle
        
        Args:
            node: Nœud à partir duquel afficher (par défaut la racine)
            level: Niveau actuel (pour l'indentation)
            prefix: Préfixe pour l'affichage
        """
        if node is None:
            node = self.root
        
        if node is not None:
            print(" " * (level * 4) + prefix + str(node.value) + 
                  f" (bin: {bin(node.value)[2:].zfill(8)})")
            
            if node.left is not None or node.right is not None:
                if node.left is not None:
                    self.display_tree(node.left, level + 1, "L--- ")
                else:
                    print(" " * ((level + 1) * 4) + "L--- None")
                
                if node.right is not None:
                    self.display_tree(node.right, level + 1, "R--- ")
                else:
                    print(" " * ((level + 1) * 4) + "R--- None")
    
    def get_tree_info(self) -> dict:
        """
        Retourne des informations sur l'arbre
        
        Returns:
            Dictionnaire contenant les statistiques de l'arbre
        """
        return {
            "size": self.size,
            "root_value": self.root.value if self.root else None,
            "height": self._calculate_height(self.root),
            "nb_bits": self.nb_bits,
            "is_empty": self.root is None
        }
    
    def _calculate_height(self, node: TreeNode) -> int:
        """Calcule la hauteur de l'arbre récursivement"""
        if node is None:
            return 0
        
        left_height = self._calculate_height(node.left)
        right_height = self._calculate_height(node.right)
        
        return 1 + max(left_height, right_height)
    
    def plot_tree(self, figsize=(12, 8), show_binary=True, save_path=None):
        """
        Affiche l'arbre sous forme graphique
        
        Args:
            figsize: Taille de la figure
            show_binary: Afficher les représentations binaires
            save_path: Chemin pour sauvegarder l'image
        """
        if self.root is None:
            print("Arbre vide")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calcule les positions des nœuds
        positions = {}
        self._calculate_positions(self.root, 0, 0, 4.0, positions)
        
        # Dessine les arêtes
        self._draw_edges(ax, self.root, positions)
        
        # Dessine les nœuds
        self._draw_nodes(ax, positions, show_binary)
        
        # Configuration de l'affichage
        ax.set_xlim(-5, 5)
        ax.set_ylim(-self._calculate_height(self.root) - 0.5, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.title(f"Arbre Binaire avec Logique de Bits\nTaille: {self.size} nœuds", 
                 fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _calculate_positions(self, node: TreeNode, x: float, y: float, 
                           spacing: float, positions: dict):
        """
        Calcule les positions des nœuds pour l'affichage graphique
        
        Args:
            node: Nœud actuel
            x, y: Position du nœud
            spacing: Espacement horizontal
            positions: Dictionnaire des positions
        """
        if node is None:
            return
        
        positions[node] = (x, y)
        
        # Calcule l'espacement pour les enfants
        new_spacing = spacing * 0.6
        
        if node.left:
            self._calculate_positions(node.left, x - spacing, y - 1, 
                                    new_spacing, positions)
        
        if node.right:
            self._calculate_positions(node.right, x + spacing, y - 1, 
                                    new_spacing, positions)
    
    def _draw_edges(self, ax, node: TreeNode, positions: dict):
        """
        Dessine TOUS les arêtes de l'arbre, y compris les liens manquants
        
        Args:
            ax: Axes matplotlib
            node: Nœud actuel
            positions: Dictionnaire des positions
        """
        if node is None:
            return
        
        x, y = positions[node]
        
        # Dessine l'arête vers l'enfant gauche (bit = 0)
        if node.left:
            x_left, y_left = positions[node.left]
            # Ligne solide pour les connexions existantes
            ax.plot([x, x_left], [y, y_left], 'k-', linewidth=2, alpha=0.8)
            
            # Étiquette "0" sur l'arête
            mid_x, mid_y = (x + x_left) / 2, (y + y_left) / 2
            ax.text(mid_x - 0.15, mid_y + 0.1, '0', fontsize=12, 
                   fontweight='bold', color='red',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                           edgecolor="red", alpha=0.9))
            
            # Récursion pour l'enfant gauche
            self._draw_edges(ax, node.left, positions)
        else:
            # Dessine une ligne pointillée pour montrer où serait l'enfant gauche
            spacing = 1.0  # Espacement approximatif
            x_left = x - spacing
            y_left = y - 1
            ax.plot([x, x_left], [y, y_left], 'r--', linewidth=1, alpha=0.4)
            
            # Étiquette "0" pour le lien manquant
            mid_x, mid_y = (x + x_left) / 2, (y + y_left) / 2
            ax.text(mid_x - 0.15, mid_y + 0.1, '0', fontsize=10, 
                   fontweight='bold', color='red', alpha=0.5,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", 
                           edgecolor="red", alpha=0.5))
        
        # Dessine l'arête vers l'enfant droit (bit = 1)
        if node.right:
            x_right, y_right = positions[node.right]
            # Ligne solide pour les connexions existantes
            ax.plot([x, x_right], [y, y_right], 'k-', linewidth=2, alpha=0.8)
            
            # Étiquette "1" sur l'arête
            mid_x, mid_y = (x + x_right) / 2, (y + y_right) / 2
            ax.text(mid_x + 0.15, mid_y + 0.1, '1', fontsize=12, 
                   fontweight='bold', color='blue',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                           edgecolor="blue", alpha=0.9))
            
            # Récursion pour l'enfant droit
            self._draw_edges(ax, node.right, positions)
        else:
            # Dessine une ligne pointillée pour montrer où serait l'enfant droit
            spacing = 1.0  # Espacement approximatif
            x_right = x + spacing
            y_right = y - 1
            ax.plot([x, x_right], [y, y_right], 'b--', linewidth=1, alpha=0.4)
            
            # Étiquette "1" pour le lien manquant
            mid_x, mid_y = (x + x_right) / 2, (y + y_right) / 2
            ax.text(mid_x + 0.15, mid_y + 0.1, '1', fontsize=10, 
                   fontweight='bold', color='blue', alpha=0.5,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", 
                           edgecolor="blue", alpha=0.5))
    
    def plot_tree_complete(self, figsize=(14, 10), show_binary=True, show_missing_links=True, save_path=None):
        """
        Affiche l'arbre avec TOUS les liens possibles (même ceux manquants)
        
        Args:
            figsize: Taille de la figure
            show_binary: Afficher les représentations binaires
            show_missing_links: Afficher les liens manquants en pointillés
            save_path: Chemin pour sauvegarder l'image
        """
        if self.root is None:
            print("Arbre vide")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calcule les positions avec un espacement plus large
        positions = {}
        max_width = max(6, self.size * 1.2)
        self._calculate_positions_complete(self.root, 0, 0, max_width / 3, positions)
        
        # Dessine les arêtes (avec ou sans liens manquants)
        if show_missing_links:
            self._draw_edges_complete(ax, self.root, positions, max_depth=3)
        else:
            self._draw_edges(ax, self.root, positions)
        
        # Dessine les nœuds
        self._draw_nodes_enhanced(ax, positions, show_binary)
        
        # Configuration de l'affichage
        height = self._calculate_height(self.root)
        ax.set_xlim(-max_width/2 - 1, max_width/2 + 1)
        ax.set_ylim(-height - 1, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.title(f"Arbre Binaire avec Logique de Bits (Complet)\nTaille: {self.size} nœuds", 
                 fontsize=16, fontweight='bold')
        
        # Légende
        if show_missing_links:
            legend_elements = [
                plt.Line2D([0], [0], color='black', linewidth=2, label='Liens existants'),
                plt.Line2D([0], [0], color='red', linestyle='--', alpha=0.4, label='Liens gauche manquants (0)'),
                plt.Line2D([0], [0], color='blue', linestyle='--', alpha=0.4, label='Liens droite manquants (1)')
            ]
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _calculate_positions_complete(self, node: TreeNode, x: float, y: float, 
                                    spacing: float, positions: dict):
        """
        Calcule les positions avec espacement adaptatif pour voir tous les liens
        """
        if node is None:
            return
        
        positions[node] = (x, y)
        
        # Espacement adaptatif selon le niveau
        new_spacing = max(spacing * 0.7, 0.8)  # Minimum pour éviter le chevauchement
        
        if node.left:
            self._calculate_positions_complete(node.left, x - spacing, y - 1.2, 
                                             new_spacing, positions)
        
        if node.right:
            self._calculate_positions_complete(node.right, x + spacing, y - 1.2, 
                                             new_spacing, positions)
    
    def _draw_edges_complete(self, ax, node: TreeNode, positions: dict, max_depth: int):
        """
        Dessine TOUS les liens possibles jusqu'à une certaine profondeur
        """
        if node is None or max_depth <= 0:
            return
        
        x, y = positions[node]
        spacing = 1.5 / (max_depth + 1)  # Espacement adaptatif
        
        # Liens vers enfant gauche (bit = 0)
        if node.left:
            x_left, y_left = positions[node.left]
            ax.plot([x, x_left], [y, y_left], 'k-', linewidth=3, alpha=0.8)
            mid_x, mid_y = (x + x_left) / 2, (y + y_left) / 2
            ax.text(mid_x - 0.2, mid_y + 0.15, '0', fontsize=14, 
                   fontweight='bold', color='red',
                   bbox=dict(boxstyle="round,pad=0.4", facecolor="white", 
                           edgecolor="red", alpha=0.9, linewidth=2))
            self._draw_edges_complete(ax, node.left, positions, max_depth - 1)
        else:
            # Lien manquant à gauche
            x_left = x - spacing * 2
            y_left = y - 1.2
            ax.plot([x, x_left], [y, y_left], 'r:', linewidth=2, alpha=0.6)
            mid_x, mid_y = (x + x_left) / 2, (y + y_left) / 2
            ax.text(mid_x - 0.2, mid_y + 0.15, '0', fontsize=12, 
                   fontweight='bold', color='red', alpha=0.6,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                           edgecolor="red", alpha=0.6))
        
        # Liens vers enfant droit (bit = 1)
        if node.right:
            x_right, y_right = positions[node.right]
            ax.plot([x, x_right], [y, y_right], 'k-', linewidth=3, alpha=0.8)
            mid_x, mid_y = (x + x_right) / 2, (y + y_right) / 2
            ax.text(mid_x + 0.2, mid_y + 0.15, '1', fontsize=14, 
                   fontweight='bold', color='blue',
                   bbox=dict(boxstyle="round,pad=0.4", facecolor="white", 
                           edgecolor="blue", alpha=0.9, linewidth=2))
            self._draw_edges_complete(ax, node.right, positions, max_depth - 1)
        else:
            # Lien manquant à droite
            x_right = x + spacing * 2
            y_right = y - 1.2
            ax.plot([x, x_right], [y, y_right], 'b:', linewidth=2, alpha=0.6)
            mid_x, mid_y = (x + x_right) / 2, (y + y_right) / 2
            ax.text(mid_x + 0.2, mid_y + 0.15, '1', fontsize=12, 
                   fontweight='bold', color='blue', alpha=0.6,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                           edgecolor="blue", alpha=0.6))
    
    def _draw_nodes_enhanced(self, ax, positions: dict, show_binary: bool):
        """
        Dessine les nœuds avec un style amélioré
        """
        for node, (x, y) in positions.items():
            # Couleur selon la valeur
            if node.value == 0:
                color = 'lightgreen'
                edge_color = 'darkgreen'
            elif node == self.root:
                color = 'gold'
                edge_color = 'darkorange'
            else:
                # Couleur selon la parité
                color = 'lightblue' if node.value % 2 == 0 else 'lightcoral'
                edge_color = 'darkblue' if node.value % 2 == 0 else 'darkred'
            
            # Dessine le cercle du nœud (plus grand)
            circle = plt.Circle((x, y), 0.4, color=color, 
                              edgecolor=edge_color, linewidth=2.5)
            ax.add_patch(circle)
            
            # Ajoute le texte
            if show_binary:
                binary_repr = bin(node.value)[2:].zfill(8)
                text = f"{node.value}\n{binary_repr}"
                fontsize = 9
            else:
                text = str(node.value)
                fontsize = 14
            
            ax.text(x, y, text, ha='center', va='center', 
                   fontsize=fontsize, fontweight='bold')
    
    def _draw_nodes(self, ax, positions: dict, show_binary: bool):
        """
        Dessine les nœuds de l'arbre
        
        Args:
            ax: Axes matplotlib
            positions: Dictionnaire des positions
            show_binary: Afficher les représentations binaires
        """
        for node, (x, y) in positions.items():
            # Couleur selon la valeur
            if node.value == 0:
                color = 'lightgreen'
                edge_color = 'darkgreen'
            elif node.value == 1:
                color = 'lightblue'
                edge_color = 'darkblue'
            else:
                color = 'lightyellow'
                edge_color = 'orange'
            
            # Dessine le cercle du nœud
            circle = plt.Circle((x, y), 0.3, color=color, 
                              edgecolor=edge_color, linewidth=2)
            ax.add_patch(circle)
            
            # Ajoute le texte
            if show_binary:
                binary_repr = bin(node.value)[2:].zfill(8)
                text = f"{node.value}\n{binary_repr}"
                fontsize = 8
            else:
                text = str(node.value)
                fontsize = 12
            
            ax.text(x, y, text, ha='center', va='center', 
                   fontsize=fontsize, fontweight='bold')
    
    def plot_tree_with_operations(self, operations: List[tuple], figsize=(15, 10)):
        """
        Affiche l'évolution de l'arbre au fur et à mesure des opérations
        
        Args:
            operations: Liste de tuples (operation, value) où operation est 'insert' ou 'delete'
            figsize: Taille de la figure
        """
        n_ops = len(operations)
        cols = min(4, n_ops + 1)  # Maximum 4 colonnes
        rows = (n_ops + 1 + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        # État initial
        self._plot_single_tree(axes[0], "État initial", show_binary=False)
        
        # Applique les opérations une par une
        for i, (operation, value) in enumerate(operations):
            if operation == 'insert':
                self.insert(value)
                title = f"Après insertion de {value}"
            elif operation == 'delete':
                self.delete(value)
                title = f"Après suppression de {value}"
            else:
                continue
            
            if i + 1 < len(axes):
                self._plot_single_tree(axes[i + 1], title, show_binary=False)
        
        # Cache les axes non utilisés
        for j in range(len(operations) + 1, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_single_tree(self, ax, title: str, show_binary: bool = False):
        """
        Dessine l'arbre sur un seul axe
        
        Args:
            ax: Axe matplotlib
            title: Titre du graphique
            show_binary: Afficher les représentations binaires
        """
        ax.clear()
        
        if self.root is None:
            ax.text(0.5, 0.5, "Arbre vide", ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(title)
            ax.axis('off')
            return
        
        # Calcule les positions
        positions = {}
        max_width = max(4, self.size * 0.8)
        self._calculate_positions(self.root, 0, 0, max_width / 4, positions)
        
        # Dessine les arêtes et nœuds
        self._draw_edges(ax, self.root, positions)
        self._draw_nodes(ax, positions, show_binary)
        
        # Configuration de l'axe
        height = self._calculate_height(self.root)
        ax.set_xlim(-max_width/2, max_width/2)
        ax.set_ylim(-height - 0.5, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=10, fontweight='bold')
    
    def compare_tree_structures(self, other_tree: 'BinaryTreeBitLogic', figsize=(15, 6)):
        """
        Compare deux arbres côte à côte
        
        Args:
            other_tree: Autre arbre à comparer
            figsize: Taille de la figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Arbre actuel
        self._plot_single_tree(ax1, f"Arbre 1 (Taille: {self.size})", show_binary=True)
        
        # Autre arbre
        other_tree._plot_single_tree(ax2, f"Arbre 2 (Taille: {other_tree.size})", show_binary=True)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_bit_patterns(self, figsize=(12, 8)):
        """
        Analyse et affiche les patterns de bits dans l'arbre
        
        Args:
            figsize: Taille de la figure
        """
        if self.root is None:
            print("Arbre vide")
            return
        
        # Collecte toutes les valeurs et leurs représentations binaires
        values = self.preorder_traversal()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Graphique 1: Représentation binaire en heatmap
        binary_matrix = []
        labels = []
        
        for value in values:
            binary_str = bin(value)[2:].zfill(8)
            binary_list = [int(bit) for bit in binary_str]
            binary_matrix.append(binary_list)
            labels.append(f"{value}")
        
        binary_array = np.array(binary_matrix)
        
        im1 = ax1.imshow(binary_array, cmap='Blues', aspect='auto')
        ax1.set_yticks(range(len(values)))
        ax1.set_yticklabels(labels)
        ax1.set_xticks(range(8))
        ax1.set_xticklabels([f"Bit {i}" for i in range(7, -1, -1)])
        ax1.set_title("Représentations Binaires des Nœuds")
        ax1.set_xlabel("Position du bit (MSB à gauche)")
        
        # Ajoute une colorbar
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Graphique 2: Distribution des valeurs
        ax2.bar(range(len(values)), values, color='skyblue', edgecolor='navy')
        ax2.set_xlabel("Index du nœud (parcours préordre)")
        ax2.set_ylabel("Valeur")
        ax2.set_title("Distribution des Valeurs dans l'Arbre")
        ax2.grid(True, alpha=0.3)
        
        # Ajoute les valeurs sur les barres
        for i, v in enumerate(values):
            ax2.text(i, v + max(values) * 0.01, str(v), 
                    ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.show()


# ============== TESTS ET DÉMONSTRATIONS ==============

def test_basic_operations():
    """Test des opérations de base"""
    print("=== Test des opérations de base ===")
    
    tree = BinaryTreeBitLogic()
    
    # Test d'insertion
    values = [1, 2, 3, 4, 5, 8, 10, 15]
    print(f"Insertion des valeurs: {values}")
    
    for value in values:
        success = tree.insert(value)
        print(f"  Insertion de {value} ({bin(value)[2:].zfill(8)}): {'Succès' if success else 'Échec'}")
    
    print(f"\nTaille de l'arbre: {tree.size}")
    print("\nStructure de l'arbre:")
    tree.display_tree()
    
    # Test de recherche
    print(f"\n=== Test de recherche ===")
    search_values = [5, 7, 15, 0]
    for value in search_values:
        node = tree.search(value)
        print(f"Recherche de {value}: {'Trouvé' if node else 'Non trouvé'}")
    
    # Test de suppression
    print(f"\n=== Test de suppression ===")
    delete_values = [3, 8]
    for value in delete_values:
        success = tree.delete(value)
        print(f"Suppression de {value}: {'Succès' if success else 'Échec'}")
        print(f"Nouvelle taille: {tree.size}")
        tree.display_tree()
        print()


def test_bit_logic():
    """Test de la logique de bits"""
    print("\n=== Test de la logique de bits ===")
    
    tree = BinaryTreeBitLogic()
    
    # Test avec des valeurs qui illustrent la logique de bits
    test_values = [
        (0, "00000000"),  # Racine
        (1, "00000001"),  # Diffère au bit 0, bit = 1 -> droite
        (2, "00000010"),  # Diffère au bit 1, bit = 1 -> droite
        (4, "00000100"),  # Diffère au bit 2, bit = 1 -> droite
        (3, "00000011"),  # Diffère au bit 1, bit = 1 -> droite, puis bit 0 = 1
    ]
    
    for value, binary in test_values:
        if value != 0:  # La racine est déjà présente
            tree.insert(value)
        print(f"Valeur {value} ({binary}) insérée")
    
    print("\nArbre résultant:")
    tree.display_tree()
    
    print(f"\nParcours inordre: {tree.inorder_traversal()}")
    print(f"Parcours préordre: {tree.preorder_traversal()}")


def test_complex_example():
    """Test avec un exemple plus complexe"""
    print("\n=== Test complexe ===")
    
    tree = BinaryTreeBitLogic()
    
    # Insertion de plusieurs valeurs
    values = [7, 3, 11, 1, 5, 9, 13, 2, 4, 6, 8, 10, 12, 14]
    
    print(f"Insertion séquentielle de: {values}")
    for value in values:
        tree.insert(value)
    
    print(f"\nArbre après insertion:")
    tree.display_tree()
    
    print(f"\nInformations sur l'arbre:")
    info = tree.get_tree_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test de suppression d'un nœud avec deux enfants
    print(f"\n=== Suppression de nœuds ===")
    tree.delete(7)
    print("Après suppression de 7:")
    tree.display_tree()


def test_graphical_display():
    """Test de l'affichage graphique"""
    print("=== Test de l'affichage graphique ===")
    
    tree = BinaryTreeBitLogic()
    
    # Insertion de valeurs pour créer un arbre intéressant
    values = [1, 2, 3, 4, 5, 8, 10, 15, 7, 12]
    
    print(f"Insertion des valeurs: {values}")
    for value in values:
        tree.insert(value)
    
    # Affichage graphique simple
    print("\n1. Affichage graphique de l'arbre:")
    tree.plot_tree(show_binary=True)
    
    # Analyse des patterns de bits
    print("\n2. Analyse des patterns de bits:")
    tree.analyze_bit_patterns()


def test_tree_evolution():
    """Test de l'évolution de l'arbre"""
    print("\n=== Test de l'évolution de l'arbre ===")
    
    tree = BinaryTreeBitLogic()
    
    # Définit une séquence d'opérations
    operations = [
        ('insert', 1),
        ('insert', 2),
        ('insert', 3),
        ('insert', 4),
        ('delete', 2),
        ('insert', 8),
        ('insert', 5)
    ]
    
    print("Évolution de l'arbre au fur et à mesure des opérations:")
    tree.plot_tree_with_operations(operations)


def test_tree_comparison():
    """Test de comparaison entre deux arbres"""
    print("\n=== Test de comparaison d'arbres ===")
    
    # Premier arbre
    tree1 = BinaryTreeBitLogic()
    values1 = [1, 2, 4, 8]
    for v in values1:
        tree1.insert(v)
    
    # Deuxième arbre
    tree2 = BinaryTreeBitLogic()
    values2 = [3, 6, 9, 12]
    for v in values2:
        tree2.insert(v)
    
    print("Comparaison de deux arbres différents:")
    tree1.compare_tree_structures(tree2)


def interactive_tree_builder():
    """Construction interactive d'un arbre"""
    print("\n=== Construction interactive d'arbre ===")
    
    tree = BinaryTreeBitLogic()
    
    print("Construction interactive d'un arbre binaire")
    print("Commandes disponibles:")
    print("  i <valeur> : insérer une valeur")
    print("  d <valeur> : supprimer une valeur")
    print("  s <valeur> : rechercher une valeur")
    print("  p : afficher l'arbre graphiquement")
    print("  t : afficher l'arbre en mode texte")
    print("  a : analyser les patterns de bits")
    print("  q : quitter")
    
    while True:
        try:
            command = input("\nCommande: ").strip().split()
            
            if not command:
                continue
            
            if command[0] == 'q':
                break
            elif command[0] == 'i' and len(command) == 2:
                value = int(command[1])
                success = tree.insert(value)
                print(f"Insertion de {value}: {'Succès' if success else 'Échec (déjà présent)'}")
            
            elif command[0] == 'd' and len(command) == 2:
                value = int(command[1])
                if value == 0:
                    print("Impossible de supprimer la racine (0)")
                else:
                    success = tree.delete(value)
                    print(f"Suppression de {value}: {'Succès' if success else 'Échec (non trouvé)'}")
            
            elif command[0] == 's' and len(command) == 2:
                value = int(command[1])
                node = tree.search(value)
                print(f"Recherche de {value}: {'Trouvé' if node else 'Non trouvé'}")
            
            elif command[0] == 'p':
                print("Affichage graphique de l'arbre:")
                tree.plot_tree(show_binary=True)
            
            elif command[0] == 't':
                print("Affichage textuel de l'arbre:")
                tree.display_tree()
            
            elif command[0] == 'a':
                print("Analyse des patterns de bits:")
                tree.analyze_bit_patterns()
            
            else:
                print("Commande non reconnue. Utilisez 'q' pour quitter.")
        
        except ValueError:
            print("Erreur: Veuillez entrer un nombre entier valide.")
        except Exception as e:
            print(f"Erreur: {e}")
    
    print("Au revoir!")


def test_basic_operations():
    """Test des opérations de base avec première donnée comme racine"""
    print("=== Test des opérations de base (première donnée = racine) ===")
    
    tree = BinaryTreeBitLogic()
    
    # L'arbre est initialement vide
    print("Arbre initial:")
    info = tree.get_tree_info()
    print(f"  Taille: {info['size']}, Racine: {info['root_value']}, Vide: {info['is_empty']}")
    
    # Test d'insertion - la première valeur devient la racine
    values = [5, 1, 2, 3, 4, 8, 10, 15]
    print(f"\nInsertion des valeurs: {values}")
    
    for i, value in enumerate(values):
        success = tree.insert(value)
        if i == 0:
            print(f"  {value} ({bin(value)[2:].zfill(8)}) devient la RACINE")
        else:
            print(f"  Insertion de {value} ({bin(value)[2:].zfill(8)}): {'Succès' if success else 'Échec'}")
    
    print(f"\nTaille de l'arbre: {tree.size}")
    print(f"Racine de l'arbre: {tree.root.value}")
    print("\nStructure de l'arbre:")
    tree.display_tree()
    
    # Test de recherche
    print(f"\n=== Test de recherche ===")
    search_values = [5, 7, 15, 1]
    for value in search_values:
        node = tree.search(value)
        print(f"Recherche de {value}: {'Trouvé' if node else 'Non trouvé'}")
    
    # Test de suppression (y compris la racine)
    print(f"\n=== Test de suppression ===")
    delete_values = [3, 8, 5]  # 5 est la racine
    for value in delete_values:
        print(f"\nAvant suppression de {value}:")
        print(f"  Racine actuelle: {tree.root.value if tree.root else 'Aucune'}")
        
        success = tree.delete(value)
        print(f"Suppression de {value}: {'Succès' if success else 'Échec'}")
        print(f"  Nouvelle taille: {tree.size}")
        print(f"  Nouvelle racine: {tree.root.value if tree.root else 'Aucune'}")
        
        if tree.root:
            tree.display_tree()
        print()


def test_different_root_values():
    """Test avec différentes valeurs de racine"""
    print("\n=== Test avec différentes valeurs de racine ===")
    
    root_values = [0, 1, 7, 15, 31]
    
    for root_val in root_values:
        print(f"\n--- Test avec racine = {root_val} ---")
        tree = BinaryTreeBitLogic()
        
        # La première insertion définit la racine
        tree.insert(root_val)
        print(f"Racine définie: {root_val} ({bin(root_val)[2:].zfill(8)})")
        
        # Ajoute quelques autres valeurs
        other_values = [root_val + 1, root_val + 2, root_val + 4]
        for val in other_values:
            if val < 32:  # Limite pour l'affichage
                tree.insert(val)
        
        print(f"Valeurs ajoutées: {tree.inorder_traversal()}")
        tree.display_tree()


def test_empty_tree_operations():
    """Test des opérations sur un arbre vide"""
    print("\n=== Test des opérations sur arbre vide ===")
    
    tree = BinaryTreeBitLogic()
    
    # Test de recherche sur arbre vide
    print("Recherche sur arbre vide:")
    result = tree.search(5)
    print(f"  Recherche de 5: {'Trouvé' if result else 'Non trouvé'}")
    
    # Test de suppression sur arbre vide
    print("Suppression sur arbre vide:")
    result = tree.delete(5)
    print(f"  Suppression de 5: {'Succès' if result else 'Échec'}")
    
    # Test de parcours sur arbre vide
    print("Parcours sur arbre vide:")
    inorder = tree.inorder_traversal()
    preorder = tree.preorder_traversal()
    print(f"  Inordre: {inorder}")
    print(f"  Préordre: {preorder}")
    
    # Première insertion
    print("\nPremière insertion (définit la racine):")
    tree.insert(10)
    print(f"  Racine: {tree.root.value}")
    print(f"  Taille: {tree.size}")


def test_root_deletion_scenarios():
    """Test de différents scénarios de suppression de racine"""
    print("\n=== Test de suppression de racine ===")
    
    scenarios = [
        ("Racine seule", [42]),
        ("Racine avec enfant gauche", [10, 5]),
        ("Racine avec enfant droit", [10, 20]),
        ("Racine avec deux enfants", [10, 5, 20, 3, 7, 15, 25])
    ]
    
    for scenario_name, values in scenarios:
        print(f"\n--- {scenario_name} ---")
        tree = BinaryTreeBitLogic()
        
        # Construction de l'arbre
        for val in values:
            tree.insert(val)
        
        print(f"Arbre avant suppression de la racine ({values[0]}):")
        tree.display_tree()
        
        # Suppression de la racine
        root_value = values[0]
        tree.delete(root_value)
        
        print(f"Arbre après suppression de la racine:")
        if tree.root:
            print(f"  Nouvelle racine: {tree.root.value}")
            tree.display_tree()
        else:
            print("  Arbre vide")
        
        print(f"  Taille: {tree.size}")


def demo_dynamic_root():
    """Démonstration de la racine dynamique"""
    print("\n=== Démonstration de la racine dynamique ===")
    
    tree = BinaryTreeBitLogic()
    
    print("Construction progressive de l'arbre:")
    values = [8, 4, 12, 2, 6, 10, 14]
    
    for i, value in enumerate(values):
        tree.insert(value)
        
        if i == 0:
            print(f"\nÉtape {i+1}: Insertion de {value} (devient la racine)")
        else:
            print(f"\nÉtape {i+1}: Insertion de {value}")
        
        print(f"  Racine: {tree.root.value}")
        print(f"  Contenu: {tree.inorder_traversal()}")
        tree.display_tree()
        print()


def test_complete_tree_display():
    """Test de l'affichage complet avec tous les liens"""
    print("=== Test de l'affichage complet ===")
    
    tree = BinaryTreeBitLogic()
    
    # Création d'un arbre avec des "trous" pour tester l'affichage
    values = [8, 4, 12, 2, 10, 15, 7]  # Arbre avec quelques nœuds manquants
    
    print(f"Construction de l'arbre avec: {values}")
    for value in values:
        tree.insert(value)
    
    print("\n1. Affichage standard:")
    tree.plot_tree(show_binary=True)
    
    print("\n2. Affichage complet avec tous les liens:")
    tree.plot_tree_complete(show_binary=True, show_missing_links=True)
    
    print("\n3. Affichage complet sans liens manquants:")
    tree.plot_tree_complete(show_binary=True, show_missing_links=False)


def test_graphical_display():
    """Test de l'affichage graphique amélioré"""
    print("=== Test de l'affichage graphique amélioré ===")
    
    tree = BinaryTreeBitLogic()
    
    # Insertion de valeurs pour créer un arbre intéressant
    values = [8, 4, 12, 2, 6, 10, 14, 1, 3, 5, 7, 9, 11, 13, 15]
    
    print(f"Insertion des valeurs: {values}")
    for value in values:
        tree.insert(value)
    
    # Affichage complet
    print("\nAffichage complet avec tous les liens:")
    tree.plot_tree_complete(figsize=(16, 12), show_binary=True, show_missing_links=True)
    
    # Analyse des patterns de bits
    print("\nAnalyse des patterns de bits:")
    tree.analyze_bit_patterns()


def interactive_tree_builder_enhanced():
    """Version améliorée du constructeur interactif"""
    print("\n=== Construction interactive d'arbre améliorée ===")
    
    tree = BinaryTreeBitLogic()
    
    print("Construction interactive d'un arbre binaire")
    print("Commandes disponibles:")
    print("  i <valeur> : insérer une valeur")
    print("  d <valeur> : supprimer une valeur")
    print("  s <valeur> : rechercher une valeur")
    print("  p : afficher l'arbre graphiquement (standard)")
    print("  pc : afficher l'arbre complet avec tous les liens")
    print("  t : afficher l'arbre en mode texte")
    print("  a : analyser les patterns de bits")
    print("  info : informations sur l'arbre")
    print("  clear : vider l'arbre")
    print("  demo : charger un exemple")
    print("  q : quitter")
    
    while True:
        try:
            command = input(f"\nArbre[{tree.size}]> ").strip().split()
            
            if not command:
                continue
            
            if command[0] == 'q':
                break
                
            elif command[0] == 'i' and len(command) == 2:
                value = int(command[1])
                success = tree.insert(value)
                if tree.size == 1:
                    print(f"√ {value} devient la racine de l'arbre")
                else:
                    print(f"√ Insertion de {value}: {'Succès' if success else 'Échec (déjà présent)'}") 
            
            elif command[0] == 'd' and len(command) == 2:
                value = int(command[1])
                if tree.root and value == tree.root.value:
                    print(f"⚠ Suppression de la racine ({value})")
                    success = tree.delete(value)
                    if success and tree.root:
                        print(f"  Nouvelle racine: {tree.root.value}")
                    elif success:
                        print("  Arbre maintenant vide")
                else:
                    success = tree.delete(value)
                    print(f"Suppression de {value}: {'Succès' if success else 'Échec (non trouvé)'}")
            
            elif command[0] == 's' and len(command) == 2:
                value = int(command[1])
                node = tree.search(value)
                print(f"Recherche de {value}: {'Trouvé' if node else 'Non trouvé'}")
            
            elif command[0] == 'p':
                if tree.root:
                    print("Affichage graphique standard:")
                    tree.plot_tree(show_binary=True)
                else:
                    print("Arbre vide")
            
            elif command[0] == 'pc':
                if tree.root:
                    print("Affichage complet avec tous les liens:")
                    tree.plot_tree_complete(show_binary=True, show_missing_links=True)
                else:
                    print("Arbre vide")
            
            elif command[0] == 't':
                if tree.root:
                    print("Affichage textuel de l'arbre:")
                    tree.display_tree()
                else:
                    print("Arbre vide")
            
            elif command[0] == 'a':
                if tree.root:
                    print("Analyse des patterns de bits:")
                    tree.analyze_bit_patterns()
                else:
                    print("Arbre vide")
            
            elif command[0] == 'info':
                info = tree.get_tree_info()
                print("Informations sur l'arbre:")
                for key, value in info.items():
                    print(f"  {key}: {value}")
                if tree.root:
                    print(f"  Contenu (inordre): {tree.inorder_traversal()}")
            
            elif command[0] == 'clear':
                tree = BinaryTreeBitLogic()
                print("Arbre vidé")
            
            elif command[0] == 'demo':
                tree = BinaryTreeBitLogic()
                demo_values = [8, 4, 12, 2, 6, 10, 14, 1, 3, 5, 7]
                for v in demo_values:
                    tree.insert(v)
                print(f"Arbre de démonstration chargé avec: {demo_values}")
                print(f"Racine: {tree.root.value}, Taille: {tree.size}")
            
            else:
                print("Commande non reconnue. Tapez 'q' pour quitter.")
        
        except ValueError:
            print("Erreur: Veuillez entrer un nombre entier valide.")
        except Exception as e:
            print(f"Erreur: {e}")
    
    print("Au revoir!")


if __name__ == "__main__":
    # Tests modifiés pour la nouvelle logique
    test_empty_tree_operations()
    test_basic_operations()
    test_different_root_values()
    test_root_deletion_scenarios()
    demo_dynamic_root()
    
    # Tests graphiques améliorés
    try:
        test_complete_tree_display()
        test_graphical_display()
        test_tree_evolution()
        test_tree_comparison()
        
        # Mode interactif amélioré
        response = input("\nVoulez-vous utiliser le mode interactif amélioré? (o/n): ")
        if response.lower() == 'o':
            interactive_tree_builder_enhanced()
    except NameError:
        print("\nFonctions graphiques non disponibles (matplotlib requis)")
    except Exception as e:
        print(f"Erreur avec les fonctions graphiques: {e}")