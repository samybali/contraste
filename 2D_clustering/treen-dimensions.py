"""
Telecom Paris : projet Contraste

Projet réalisé par : Samy Bali, Hussein El gouch, Issa Daou, Anas El Barhdadi

Encadré par Jean-Louis Dessalles
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Set, Dict
import networkx as nx
from dataclasses import dataclass, field
import matplotlib.patches as mpatches
import itertools


@dataclass
class NodeND:
    """
    Représente un nœud dans l'arbre hiérarchique n-dimensionnel.
    
    Attributes:
        pattern (List[str]): Représentation binaire avec étoiles pour les wildcards
        children (List[NodeND]): Liste des nœuds enfants
        parent (Optional[NodeND]): Référence vers le nœud parent
        is_leaf (bool): Indique si c'est une feuille (donnée originale)
        original_data (Optional[List[int]]): Données originales si c'est une feuille
        id (int): Identifiant unique du nœud
    """
    pattern: List[str]
    children: List['NodeND'] = field(default_factory=list)
    parent: Optional['NodeND'] = None
    is_leaf: bool = False
    original_data: Optional[List[int]] = None
    id: int = 0


class BinaryHierarchicalTreeND:
    """
    Implémente un arbre hiérarchique basé sur la représentation binaire en n dimensions
    avec opérateur de contraste.
    """
    
    def __init__(self, n_dims: int, n_bits: int = 8, contrast_threshold: int = 3):
        """
        Initialise l'arbre.
        
        Args:
            n_dims (int): Nombre de dimensions
            n_bits (int): Nombre de bits pour représenter chaque composante
            contrast_threshold (int): Seuil de contraste (nombre minimum de bits différents pour considérer une dimension)
        """
        self.n_dims = n_dims
        self.n_bits = n_bits
        self.contrast_threshold = contrast_threshold
        self.root = None
        self.node_counter = 0
        self.all_nodes = []
        self.pattern_to_node = {}
        
    def _to_binary(self, value: int) -> str:
        """
        Convertit un entier en représentation binaire.
        """
        return format(value, f'0{self.n_bits}b')
    
    def _hamming_distance(self, bin1: str, bin2: str) -> int:
        """
        Calcule la distance de Hamming entre deux chaînes binaires.
        """
        return sum(b1 != b2 for b1, b2 in zip(bin1, bin2))
    def _weighted_distance(self, val1: int, val2: int) -> int:
        """
        Calcule la distance pondérée entre deux valeurs.
        C'est simplement la valeur absolue de la différence.
        """
        return abs(val1 - val2)

    def _create_subsumption_pattern_with_contrast(self, patterns: List[List[str]]) -> List[str]:
        """
        Crée un pattern de subsumption en utilisant l'opérateur de contraste.
        
        Pour chaque dimension :
        - Si la distance pondérée < seuil : on garde arbitrairement une valeur (dimension similaire)
        - Si la distance pondérée >= seuil : on applique la subsumption normale
        
        Args:
            patterns: Liste des patterns à subsumer
            
        Returns:
            List[str]: Pattern de subsumption
        """
        if not patterns:
            return ["*" * self.n_bits] * self.n_dims
        
        if len(patterns) == 1:
            return patterns[0].copy()
        
        result = []
        
        # Convertir les patterns binaires en valeurs décimales pour chaque dimension
        decimal_patterns = []
        for pattern in patterns:
            decimal_pattern = []
            for dim_binary in pattern:
                decimal_pattern.append(int(dim_binary, 2))
            decimal_patterns.append(decimal_pattern)
        
        # Pour chaque dimension
        for dim in range(self.n_dims):
            # Extraire les valeurs binaires et décimales pour cette dimension
            dim_binary_values = [p[dim] for p in patterns]
            dim_decimal_values = [dp[dim] for dp in decimal_patterns]
            
            # Si toutes les valeurs sont identiques
            if all(v == dim_decimal_values[0] for v in dim_decimal_values):
                result.append(dim_binary_values[0])
                continue
            
            # Calculer la distance pondérée maximale entre toutes les paires
            max_distance = 0
            for i in range(len(dim_decimal_values)):
                for j in range(i + 1, len(dim_decimal_values)):
                    distance = self._weighted_distance(dim_decimal_values[i], dim_decimal_values[j])
                    max_distance = max(max_distance, distance)
            
            # Appliquer l'opérateur de contraste
            if max_distance < self.contrast_threshold:
                # Distance inférieure au seuil : dimension similaire
                # On garde arbitrairement la première valeur
                result.append(dim_binary_values[0])
            else:
                # Distance >= seuil : dimension contrastée
                # Appliquer la subsumption normale
                subsumption = ""
                for bit_pos in range(self.n_bits):
                    bits = [v[bit_pos] for v in dim_binary_values]
                    if all(b == bits[0] for b in bits):
                        subsumption += bits[0]
                    else:
                        # Premier bit différent trouvé, le reste devient des étoiles
                        subsumption += '*' * (self.n_bits - bit_pos)
                        break
                result.append(subsumption)
        
        return result

    def calculate_threshold_for_bits(self, num_bits_to_ignore: int) -> int:
        """
        Calcule le seuil pour ignorer un certain nombre de bits de poids faible.
        
        Args:
            num_bits_to_ignore: Nombre de bits de poids faible à ignorer
            
        Returns:
            int: Seuil correspondant
        """
        if num_bits_to_ignore <= 0:
            return 0
        if num_bits_to_ignore >= self.n_bits:
            return 2**self.n_bits - 1
        
        # Somme des puissances de 2 pour les bits à ignorer
        threshold = sum(2**i for i in range(num_bits_to_ignore))
        return threshold
    
    def _can_subsume(self, pattern: List[str], data_pattern: List[str]) -> bool:
        """
        Vérifie si un pattern peut subsumer un autre.
        """
        for p_dim, d_dim in zip(pattern, data_pattern):
            for p_bit, d_bit in zip(p_dim, d_dim):
                if p_bit != '*' and p_bit != d_bit:
                    return False
        return True
    
    def _get_or_create_node(self, pattern: List[str], is_leaf: bool, 
                           original_data: Optional[List[int]] = None) -> NodeND:
        """
        Retourne un nœud existant avec ce pattern ou en crée un nouveau.
        """
        # Pour les feuilles, on crée toujours un nouveau nœud
        if is_leaf:
            node = NodeND(
                pattern=pattern,
                is_leaf=True,
                original_data=original_data,
                id=self.node_counter
            )
            self.node_counter += 1
            self.all_nodes.append(node)
            return node
        
        # Pour les prototypes, vérifier s'il existe déjà
        pattern_tuple = tuple(pattern)  # Convertir en tuple pour le hashing
        if pattern_tuple in self.pattern_to_node:
            return self.pattern_to_node[pattern_tuple]
        
        # Créer un nouveau nœud prototype
        node = NodeND(
            pattern=pattern,
            is_leaf=False,
            id=self.node_counter
        )
        self.node_counter += 1
        self.all_nodes.append(node)
        self.pattern_to_node[pattern_tuple] = node
        
        return node
    
    def insert(self, data: List[int]):
        """
        Insère une nouvelle donnée dans l'arbre.
        
        Args:
            data: Liste de valeurs entières (une par dimension)
        """
        if len(data) != self.n_dims:
            raise ValueError(f"Les données doivent avoir {self.n_dims} dimensions")
        
        pattern = [self._to_binary(val) for val in data]
        
        if self.root is None:
            # Premier élément
            self.root = self._get_or_create_node(pattern, True, data)
        else:
            # Insérer dans l'arbre existant
            self._insert_into_tree(pattern, data)
    
    def _find_all_subsumers(self, pattern: List[str]) -> List[NodeND]:
        """
        Trouve tous les nœuds qui peuvent subsumer le pattern donné.
        """
        subsumers = []
        visited = set()  # Pour éviter les boucles infinies
        
        def search_subsumers(node: NodeND):
            # Éviter de visiter le même nœud plusieurs fois
            if node.id in visited:
                return
            visited.add(node.id)
            
            if not node.is_leaf and self._can_subsume(node.pattern, pattern):
                subsumers.append(node)
            
            for child in node.children:
                search_subsumers(child)
        
        if self.root:
            search_subsumers(self.root)
        
        # Trier par spécificité (moins d'étoiles = plus spécifique)
        subsumers.sort(key=lambda n: sum(dim.count('*') for dim in n.pattern))
        
        return subsumers

    def _reorganize_tree(self, new_node: NodeND, parent: NodeND):
        """
        Réorganise l'arbre après l'insertion d'un nœud.
        Vérifie si des enfants du parent devraient être déplacés sous le nouveau nœud.
        """
        children_to_move = []
        
        for child in parent.children[:]:  # Copie de la liste pour éviter les problèmes
            if child != new_node and child.id != new_node.id and self._can_subsume(new_node.pattern, child.pattern):
                children_to_move.append(child)
        
        # Déplacer les enfants appropriés
        for child in children_to_move:
            parent.children.remove(child)
            child.parent = new_node
            new_node.children.append(child)

    def _insert_into_tree(self, pattern: List[str], data: List[int]):
        """
        Insère une nouvelle donnée dans l'arbre existant.
        """
        # Créer le nouveau nœud feuille
        new_leaf = self._get_or_create_node(pattern, True, data)
        
        # Trouver tous les subsumeurs possibles
        subsumers = self._find_all_subsumers(pattern)
        
        if subsumers:
            # Utiliser le subsumeur le plus spécifique
            best_parent = subsumers[0]
            
            # Vérifier s'il faut créer un nœud intermédiaire
            siblings = [child for child in best_parent.children if child.is_leaf]
            
            if siblings:
                # Ne pas créer d'intermédiaire si on a déjà des données identiques
                identical_siblings = [s for s in siblings if s.pattern == pattern]
                
                if not identical_siblings:
                    # Chercher le meilleur frère pour créer un groupe
                    best_sibling = None
                    best_combined_pattern = None
                    
                    for sibling in siblings:
                        combined = self._create_subsumption_pattern_with_contrast(
                            [sibling.pattern, pattern]
                        )
                        
                        # Vérifier si le pattern combiné est valide et différent du parent
                        if (combined != best_parent.pattern and 
                            self._can_subsume(best_parent.pattern, combined)):
                            
                            # Calculer le nombre d'étoiles
                            combined_stars = sum(dim.count('*') for dim in combined)
                            
                            if best_combined_pattern is None:
                                best_sibling = sibling
                                best_combined_pattern = combined
                            else:
                                # Garder le pattern le plus spécifique
                                best_stars = sum(dim.count('*') for dim in best_combined_pattern)
                                if combined_stars < best_stars:
                                    best_sibling = sibling
                                    best_combined_pattern = combined
                    
                    if best_sibling and best_combined_pattern:
                        # Créer le nœud intermédiaire
                        intermediate = self._get_or_create_node(best_combined_pattern, False)
                        
                        # S'assurer que le nœud intermédiaire n'est pas déjà dans l'arbre
                        # avec une relation parent-enfant incorrecte
                        if intermediate.parent and intermediate.parent.id == best_parent.id:
                            # Le nœud existe déjà sous le bon parent
                            if best_sibling not in intermediate.children:
                                # Déplacer le sibling
                                if best_sibling.parent == best_parent:
                                    best_parent.children.remove(best_sibling)
                                best_sibling.parent = intermediate
                                intermediate.children.append(best_sibling)
                            
                            # Ajouter la nouvelle feuille
                            new_leaf.parent = intermediate
                            intermediate.children.append(new_leaf)
                        else:
                            # Gérer les relations existantes
                            if intermediate.parent:
                                intermediate.parent.children.remove(intermediate)
                            
                            # Établir les nouvelles relations
                            intermediate.parent = best_parent
                            if intermediate not in best_parent.children:
                                best_parent.children.append(intermediate)
                            
                            # Déplacer le sibling
                            if best_sibling.parent == best_parent:
                                best_parent.children.remove(best_sibling)
                            best_sibling.parent = intermediate
                            intermediate.children = [best_sibling, new_leaf]
                            
                            # Ajouter la nouvelle feuille
                            new_leaf.parent = intermediate
                            
                            # Réorganiser pour voir si d'autres nœuds devraient être sous intermediate
                            self._reorganize_tree(intermediate, best_parent)
                        
                        return
            
            # Ajouter directement comme enfant
            new_leaf.parent = best_parent
            best_parent.children.append(new_leaf)
            
        else:
            # Aucun subsumeur trouvé, créer un nouveau parent pour root et new_leaf
            new_root_pattern = self._create_subsumption_pattern_with_contrast(
                [self.root.pattern, pattern]
            )
            new_root = self._get_or_create_node(new_root_pattern, False)
            
            # S'assurer qu'on ne crée pas de boucle
            if new_root.id != self.root.id:
                self.root.parent = new_root
                new_leaf.parent = new_root
                new_root.children = [self.root, new_leaf]
                self.root = new_root
            else:
                # Pattern identique à la racine, ajouter directement
                new_leaf.parent = self.root
                self.root.children.append(new_leaf)

    def _verify_tree_structure(self) -> bool:
        """
        Vérifie l'intégrité de l'arbre (pas de boucles, structure cohérente).
        """
        visited = set()
        issues = []
        
        def check_node(node: NodeND, ancestors: Set[int]):
            if node.id in ancestors:
                issues.append(f"Boucle détectée: le nœud {node.id} est son propre ancêtre")
                return
            
            if node.id in visited:
                return
            
            visited.add(node.id)
            new_ancestors = ancestors | {node.id}
            
            for child in node.children:
                if child.parent != node:
                    issues.append(f"Incohérence parent-enfant: {child.id} n'a pas le bon parent")
                check_node(child, new_ancestors)
        
        if self.root:
            check_node(self.root, set())
        
        if issues:
            print("Problèmes détectés dans l'arbre:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        
        return True
    
    def calculate_contrast_score(self, node: NodeND) -> float:
        """
        Calcule un score de contraste pour un nœud.
        
        Le score est basé sur :
        - Le nombre de dimensions ignorées (étoiles complètes)
        - La profondeur dans l'arbre
        - Le nombre de données directes
        """
        if node.is_leaf:
            return 0
        
        # Compter les dimensions complètement ignorées
        ignored_dims = sum(1 for dim in node.pattern if dim == '*' * self.n_bits)
        
        # Calculer la profondeur
        depth = 0
        current = node
        while current.parent:
            depth += 1
            current = current.parent
        
        # Compter les données directes
        data_children = sum(1 for child in node.children if child.is_leaf)
        
        # Score = dimensions_ignorées + profondeur - données_directes
        score = ignored_dims + depth - data_children
        
        return score
    
    def print_tree(self, node: Optional[NodeND] = None, depth: int = 0):
        """
        Affiche l'arbre sous forme textuelle.
        """
        if node is None:
            if self.root is None:
                print("L'arbre est vide.")
                return
            node = self.root
        
        indent = "  " * depth
        pattern_str = f"({', '.join(node.pattern)})"
        
        if node.is_leaf:
            print(f"{indent}├─ {pattern_str} [Donnée: {node.original_data}]")
        else:
            contrast_score = self.calculate_contrast_score(node)
            print(f"{indent}├─ {pattern_str} [Prototype - Score: {contrast_score}]")
        
        for child in node.children:
            self.print_tree(child, depth + 1)
    
    def visualize_projections_2d(self, dims: Tuple[int, int] = (0, 1), 
                                figsize: Tuple[int, int] = (10, 8)):
        """
        Visualise une projection 2D de l'arbre sur deux dimensions spécifiques.
        
        Args:
            dims: Tuple des deux dimensions à visualiser
            figsize: Taille de la figure
        """
        if self.root is None:
            print("L'arbre est vide.")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Collecter les clusters
        clusters = {}
        for node in self.all_nodes:
            if not node.is_leaf and node != self.root:
                data_points = []
                for child in node.children:
                    if child.is_leaf:
                        # Extraire seulement les dimensions demandées
                        point_2d = (child.original_data[dims[0]], 
                                   child.original_data[dims[1]])
                        data_points.append(point_2d)
                
                if data_points:
                    clusters[node.id] = (node, data_points)
        
        # Points isolés
        isolated_points = []
        if self.root:
            for child in self.root.children:
                if child.is_leaf:
                    point_2d = (child.original_data[dims[0]], 
                               child.original_data[dims[1]])
                    isolated_points.append(point_2d)
        
        # Visualiser
        colors = plt.cm.tab10(np.linspace(0, 1, len(clusters) + len(isolated_points)))
        
        # Clusters normaux
        for idx, (node_id, (node, points)) in enumerate(clusters.items()):
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            
            pattern_str = f"Dim{dims[0]}:{node.pattern[dims[0]]}, Dim{dims[1]}:{node.pattern[dims[1]]}"
            ax.scatter(x_coords, y_coords, c=[colors[idx]], 
                      label=pattern_str, s=100, alpha=0.7)
        
        # Points isolés
        if isolated_points:
            for i, point in enumerate(isolated_points):
                ax.scatter(point[0], point[1], c=[colors[len(clusters) + i]], 
                          marker='D', s=150, label=f'Isolé {point}')
        
        ax.set_xlabel(f'Dimension {dims[0]}')
        ax.set_ylabel(f'Dimension {dims[1]}')
        ax.set_title(f'Projection 2D sur les dimensions {dims[0]} et {dims[1]}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    def visualize(self, figsize: Tuple[int, int] = (20, 12)):
        """
        Visualise l'arbre hiérarchique n-dimensionnel.
        """
        if self.root is None:
            print("L'arbre est vide.")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculer les positions
        pos = self._calculate_positions()
        
        # Créer le graphe pour les arêtes
        G = nx.DiGraph()
        
        # Vérifier que les nœuds existent
        valid_node_ids = {node.id for node in self.all_nodes}
        
        def add_edges(node: NodeND):
            if node.id not in valid_node_ids:
                return
            for child in node.children:
                if child.id in valid_node_ids:
                    G.add_edge(node.id, child.id)
                    add_edges(child)
        
        if self.root and self.root.id in valid_node_ids:
            add_edges(self.root)
        
        # Dessiner les arêtes
        if G.edges():
            valid_pos = {node_id: position for node_id, position in pos.items() 
                        if node_id in valid_node_ids}
            
            nx.draw_networkx_edges(G, valid_pos, edge_color='gray', arrows=True, 
                                arrowsize=15, ax=ax, width=2, alpha=0.6)
        
        # Dessiner les nœuds avec leurs labels
        for node in self.all_nodes:
            if node.id in pos:
                x, y = pos[node.id]
                
                # Créer le label
                if node.is_leaf:
                    # Pour les feuilles, afficher les données originales
                    label = f"Data:\n{node.original_data}"
                    color = 'lightgreen'
                    fontsize = 8
                    weight = 'bold'
                else:
                    # Pour les prototypes, afficher le pattern avec analyse
                    label = self._format_pattern_label(node)
                    color = 'lightblue'
                    fontsize = 9
                    weight = 'normal'
                    
                    # Ajouter le score de contraste
                    contrast_score = self.calculate_contrast_score(node)
                    if node == self.root:
                        score_text = "ROOT"
                    else:
                        score_text = f"Score: {contrast_score}"
                    
                    # Afficher le score au-dessus du nœud
                    ax.text(x, y + 0.4, score_text, ha='center', va='bottom',
                        fontsize=8, weight='bold', style='italic')
                
                bbox_props = dict(
                    boxstyle="round,pad=0.4", 
                    facecolor=color, 
                    edgecolor='black', 
                    linewidth=2 if node.is_leaf else 1
                )
                
                ax.text(x, y, label, ha='center', va='center',
                    bbox=bbox_props, fontsize=fontsize, 
                    family='monospace', weight=weight)
        
        # Ajuster les limites
        all_x = [pos[node.id][0] for node in self.all_nodes if node.id in pos]
        all_y = [pos[node.id][1] for node in self.all_nodes if node.id in pos]
        
        x_margin = (max(all_x) - min(all_x)) * 0.1
        y_margin = (max(all_y) - min(all_y)) * 0.1 + 1
        
        ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
        ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
        
        ax.set_title(f'Arbre Hiérarchique {self.n_dims}D avec Contraste (seuil={self.contrast_threshold})', 
                    fontsize=16)
        ax.axis('off')
        
        # Statistiques
        leaf_count = sum(1 for n in self.all_nodes if n.is_leaf)
        proto_count = sum(1 for n in self.all_nodes if not n.is_leaf)
        
        # Légende avec statistiques
        leaf_patch = mpatches.Patch(color='lightgreen', label=f'Données ({leaf_count})')
        node_patch = mpatches.Patch(color='lightblue', label=f'Prototypes ({proto_count})')
        info_text = f'Dimensions: {self.n_dims}, Bits: {self.n_bits}'
        
        ax.legend(handles=[leaf_patch, node_patch], 
                title=info_text,
                loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plt.show()

    def analyze_pattern_contrasts(self, node: NodeND) -> Dict[int, str]:
        """
        Analyse quelles dimensions sont ignorées (complètement en étoiles) dans un pattern.
        
        Returns:
            Dict[int, str]: {dimension: 'ignored'|'partial'|'specific'}
        """
        analysis = {}
        
        for dim_idx, dim_pattern in enumerate(node.pattern):
            star_count = dim_pattern.count('*')
            
            if star_count == self.n_bits:
                analysis[dim_idx] = 'ignored'
            elif star_count > 0:
                analysis[dim_idx] = 'partial'
            else:
                analysis[dim_idx] = 'specific'
        
        return analysis
    def _format_pattern_label(self, node: NodeND) -> str:
        """
        Formate le label d'un pattern pour l'affichage.
        Montre quelles dimensions sont ignorées.
        """
        lines = []
        
        # Analyser le pattern
        analysis = self.analyze_pattern_contrasts(node)
        
        # Grouper les dimensions par type
        ignored_dims = []
        partial_dims = []
        specific_dims = []
        
        for dim, status in analysis.items():
            if status == 'ignored':
                ignored_dims.append(dim)
            elif status == 'partial':
                partial_dims.append(dim)
            else:
                specific_dims.append(dim)
        
        # Créer un affichage compact
        if len(ignored_dims) > 2:
            # Si trop de dimensions, afficher de manière compacte
            lines.append(f"Pattern {node.id}:")
            lines.append(f"Ignored dims: {ignored_dims}")
            
            # Afficher quelques dimensions non ignorées
            for dim in specific_dims[:2]:
                lines.append(f"D{dim}: {node.pattern[dim]}")
            
            if len(specific_dims) > 2:
                lines.append(f"... +{len(specific_dims)-2} dims")
        else:
            # Afficher toutes les dimensions
            lines.append(f"Pattern {node.id}:")
            for dim in range(min(5, self.n_dims)):  # Limiter à 5 dimensions pour l'affichage
                if dim in ignored_dims:
                    lines.append(f"D{dim}: ****")
                else:
                    lines.append(f"D{dim}: {node.pattern[dim]}")
            
            if self.n_dims > 5:
                lines.append(f"... +{self.n_dims-5} dims")
        
        # Ajouter le nombre d'enfants
        data_children = sum(1 for child in node.children if child.is_leaf)
        proto_children = sum(1 for child in node.children if not child.is_leaf)
        lines.append(f"Children: {data_children}D, {proto_children}P")
        
        return '\n'.join(lines)

    def _calculate_positions(self) -> Dict[int, Tuple[float, float]]:
        """
        Calcule les positions des nœuds pour la visualisation.
        """
        positions = {}
        levels = {}
        
        # Première passe : déterminer les niveaux
        def assign_levels(node: NodeND, level: int = 0):
            if level not in levels:
                levels[level] = []
            levels[level].append(node)
            
            for child in node.children:
                assign_levels(child, level + 1)
        
        if self.root:
            assign_levels(self.root)
        
        # Calculer la largeur nécessaire
        leaf_count = sum(1 for node in self.all_nodes if node.is_leaf)
        base_width = max(leaf_count * 3, 20)
        
        # Deuxième passe : assigner les positions
        def calculate_subtree_width(node: NodeND) -> int:
            if node.is_leaf:
                return 1
            return max(1, sum(calculate_subtree_width(child) for child in node.children))
        
        def assign_positions_improved(node: NodeND, x: float = 0, y: float = 0, 
                                    allocated_width: float = None):
            positions[node.id] = (x, y)
            
            if node.children:
                subtree_widths = [calculate_subtree_width(child) for child in node.children]
                total_width = sum(subtree_widths)
                
                if allocated_width is None:
                    allocated_width = base_width
                
                current_x = x - allocated_width / 2
                for i, child in enumerate(node.children):
                    child_width = (subtree_widths[i] / total_width) * allocated_width
                    child_x = current_x + child_width / 2
                    
                    assign_positions_improved(
                        child, 
                        child_x, 
                        y - 3,
                        child_width * 0.9
                    )
                    
                    current_x += child_width
        
        if self.root:
            assign_positions_improved(self.root)
        
        return positions


def test_contrast_examples():
    """
    Teste l'opérateur de contraste avec plusieurs exemples.
    """
    print("=== Tests de l'opérateur de contraste ===\n")
    
    # Créer un arbre avec seuil = 3
    tree = BinaryHierarchicalTreeND(n_dims=5, n_bits=4, contrast_threshold=3)
    
    # Exemple 1 : votre exemple original
    print("EXEMPLE 1 : Cas avec dimension complètement différente")
    data1 = [0b1101, 0b1100, 0b1110, 0b0101, 0b1101]  # (13, 12, 14, 5, 13)
    data2 = [0b1110, 0b1101, 0b1111, 0b1101, 0b1110]  # (14, 13, 15, 13, 14)
    tree.insert(data1)
    tree.insert(data2)
    tree.visualize()
    
    # Exemple 2 : Cas où la différence est au milieu
    print("\n\nEXEMPLE 2 : Cas avec différence au 3ème bit")
    data3 = [0b1101, 0b1010, 0b0000, 0b1111, 0b1000]  # (13, 10, 0, 15, 8)
    data4 = [0b1101, 0b0010, 0b0000, 0b1111, 0b1000]  # (13, 2, 0, 15, 8)
    # Pour dimension 1: 1010 vs 0010 -> différence au bit 0, donc pattern sera ****
    
    tree.insert(data3)
    tree.insert(data4)
    tree.visualize()
    
    # Exemple 3 : Cas où la différence est à la fin
    print("\n\nEXEMPLE 3 : Cas avec différence au dernier bit")
    data5 = [0b1110, 0b1100, 0b1111, 0b0000, 0b1010]  # (14, 12, 15, 0, 10)
    data6 = [0b1111, 0b1100, 0b1111, 0b0000, 0b1010]  # (15, 12, 15, 0, 10)
    # Pour dimension 0: 1110 vs 1111 -> différence au bit 3, donc pattern sera 111*
    tree.insert(data5)
    tree.insert(data6)
    tree.visualize()
    
    return tree


# Fonction de test
def test_nd_clustering():
    """
    Teste le clustering en n dimensions avec contraste.
    """
    print("=== Test du clustering n-dimensionnel avec contraste ===\n")
    
    # Créer un arbre 5D avec seuil de contraste = 3
    tree = BinaryHierarchicalTreeND(n_dims=5, n_bits=4, contrast_threshold=3)
    
    # Données de test
    test_data = [
        [0b1111, 0b1101, 0b1101, 0b0111, 0b1101],  # (15, 13, 13, 7, 13)
        [0b1110, 0b1110, 0b1100, 0b1011, 0b1101],  # (14, 14, 12, 11, 13)
        [0b1111, 0b1100, 0b1101, 0b0110, 0b1101],  # (15, 12, 13, 6, 13)
        [0b0001, 0b0010, 0b0011, 0b0100, 0b0101],  # (1, 2, 3, 4, 5)
        [0b0001, 0b0010, 0b0011, 0b1100, 0b0101],  # (1, 2, 3, 12, 5)
    ]
    
    for i, data in enumerate(test_data):
        print(f"\nInsertion {i+1}: {data}")
        pattern = [format(val, '04b') for val in data]
        print(f"  Pattern binaire: {pattern}")
        tree.insert(data)
    
    print("\n\nStructure de l'arbre:")
    tree.print_tree()
    
    # Visualiser des projections 2D
    print("\n\nVisualisations 2D:")
    
    tree.visualize()  # Visualisation complète de l'arbre
    
    return tree


if __name__ == "__main__":
    test_nd_clustering()
    test_contrast_examples()