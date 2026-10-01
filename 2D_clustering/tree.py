import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches
from typing import List, Tuple, Optional, Set, Dict
import networkx as nx
from dataclasses import dataclass, field
import itertools


@dataclass
class Node:
    """
    Représente un nœud dans l'arbre hiérarchique.
    
    Attributes:
        pattern (Tuple[str, str]): Représentation binaire avec étoiles pour les wildcards
        children (List[Node]): Liste des nœuds enfants
        parent (Optional[Node]): Référence vers le nœud parent
        is_leaf (bool): Indique si c'est une feuille (donnée originale)
        original_data (Optional[Tuple[int, int]]): Données originales si c'est une feuille
        id (int): Identifiant unique du nœud
    """
    pattern: Tuple[str, str]
    children: List['Node'] = field(default_factory=list)
    parent: Optional['Node'] = None
    is_leaf: bool = False
    original_data: Optional[Tuple[int, int]] = None
    id: int = 0


class BinaryHierarchicalTree:
    """
    Implémente un arbre hiérarchique basé sur la représentation binaire.
    
    Garantit l'unicité des patterns dans l'arbre.
    """
    
    def __init__(self, n_bits: int = 4):
        """
        Initialise l'arbre.
        
        Args:
            n_bits (int): Nombre de bits pour représenter chaque composante
        """
        self.n_bits = n_bits
        self.root = None
        self.node_counter = 0
        self.all_nodes = []
        # Dictionnaire pour garantir l'unicité des patterns
        self.pattern_to_node = {}
        
    def _to_binary(self, value: int) -> str:
        """
        Convertit un entier en représentation binaire.
        """
        return format(value, f'0{self.n_bits}b')
    
    def _can_subsume(self, pattern: Tuple[str, str], data_pattern: Tuple[str, str]) -> bool:
        """
        Vérifie si un pattern peut subsumer un autre.
        """
        for p_comp, d_comp in zip(pattern, data_pattern):
            for p_bit, d_bit in zip(p_comp, d_comp):
                if p_bit != '*' and p_bit != d_bit:
                    return False
        return True
    
    def _create_subsumption_pattern(self, patterns: List[Tuple[str, str]]) -> Tuple[str, str]:
        """
        Crée un pattern de subsumption pour plusieurs patterns.
        
        Args:
            patterns: Liste des patterns à subsumer
            
        Returns:
            Tuple[str, str]: Pattern de subsumption
        """
        if not patterns:
            return ("*" * self.n_bits, "*" * self.n_bits)
        
        result_x = ""
        result_y = ""
        
        # Pour chaque position de bit en X
        for bit_pos in range(self.n_bits):
            x_bits = [p[0][bit_pos] for p in patterns]
            
            if all(b == x_bits[0] for b in x_bits):
                result_x += x_bits[0]
            else:
                # Dès qu'on trouve une différence, le reste devient des étoiles
                result_x += '*' * (self.n_bits - bit_pos)
                break
        
        # Pour chaque position de bit en Y
        for bit_pos in range(self.n_bits):
            y_bits = [p[1][bit_pos] for p in patterns]
            
            if all(b == y_bits[0] for b in y_bits):
                result_y += y_bits[0]
            else:
                result_y += '*' * (self.n_bits - bit_pos)
                break
        
        return (result_x, result_y)
    
    def _get_or_create_node(self, pattern: Tuple[str, str], is_leaf: bool, 
                           original_data: Optional[Tuple[int, int]] = None) -> Node:
        """
        Retourne un nœud existant avec ce pattern ou en crée un nouveau.
        
        IMPORTANT: Garantit qu'un pattern n'existe qu'une seule fois dans l'arbre.
        """
        # Pour les feuilles, on crée toujours un nouveau nœud
        # car plusieurs données peuvent avoir le même pattern binaire
        if is_leaf:
            node = Node(
                pattern=pattern,
                is_leaf=True,
                original_data=original_data,
                id=self.node_counter
            )
            self.node_counter += 1
            self.all_nodes.append(node)
            return node
        
        # Pour les prototypes, vérifier s'il existe déjà
        if pattern in self.pattern_to_node:
            return self.pattern_to_node[pattern]
        
        # Créer un nouveau nœud prototype
        node = Node(
            pattern=pattern,
            is_leaf=False,
            id=self.node_counter
        )
        self.node_counter += 1
        self.all_nodes.append(node)
        self.pattern_to_node[pattern] = node
        
        return node
    
    def insert(self, data: Tuple[int, int]):
        """
        Insère une nouvelle donnée dans l'arbre.
        """
        pattern = (self._to_binary(data[0]), self._to_binary(data[1]))
        
        if self.root is None:
            # Premier élément
            self.root = self._get_or_create_node(pattern, True, data)
        else:
            # Insérer dans l'arbre existant
            self._insert_into_tree(pattern, data)
    
    def _find_all_subsumers(self, pattern: Tuple[str, str]) -> List[Node]:
        """
        Trouve tous les nœuds qui peuvent subsumer le pattern donné.
        Retourne une liste triée du plus spécifique au plus général.
        """
        subsumers = []
        
        def search_subsumers(node: Node):
            if not node.is_leaf and self._can_subsume(node.pattern, pattern):
                subsumers.append(node)
            for child in node.children:
                search_subsumers(child)
        
        if self.root:
            search_subsumers(self.root)
        
        # Trier par spécificité (moins d'étoiles = plus spécifique)
        subsumers.sort(key=lambda n: (n.pattern[0].count('*') + n.pattern[1].count('*')))
        
        return subsumers
    
    def _reorganize_tree(self, new_node: Node, parent: Node):
        """
        Réorganise l'arbre après l'insertion d'un nœud.
        Vérifie si des enfants du parent devraient être déplacés sous le nouveau nœud.
        """
        children_to_move = []
        
        for child in parent.children:
            if child != new_node and self._can_subsume(new_node.pattern, child.pattern):
                children_to_move.append(child)
        
        # Déplacer les enfants appropriés
        for child in children_to_move:
            parent.children.remove(child)
            child.parent = new_node
            new_node.children.append(child)
    
    def _should_create_intermediate_node(self, pattern1: Tuple[str, str], 
                                        pattern2: Tuple[str, str], 
                                        parent_pattern: Tuple[str, str]) -> bool:
        """
        Détermine si un nœud intermédiaire doit être créé entre deux patterns.
        
        Retourne False si :
        - Les patterns sont identiques
        - La différence avec le parent est trop faible (< 2 étoiles)
        - Le pattern intermédiaire serait trop général
        """
        # Si les patterns sont identiques, pas besoin d'intermédiaire
        if pattern1 == pattern2:
            return False
        
        # Calculer le pattern de subsumption
        combined = self._create_subsumption_pattern([pattern1, pattern2])
        
        # Si le pattern combiné est le même que le parent, pas d'intermédiaire
        if combined == parent_pattern:
            return False
        
        # Calculer la différence en nombre d'étoiles
        parent_stars = parent_pattern[0].count('*') + parent_pattern[1].count('*')
        combined_stars = combined[0].count('*') + combined[1].count('*')
        
        # Créer un intermédiaire seulement si la différence est significative
        # (au moins 2 étoiles de différence)
        return parent_stars - combined_stars >= 2

    
    def _insert_into_tree(self, pattern: Tuple[str, str], data: Tuple[int, int]):
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
                        if self._should_create_intermediate_node(
                            sibling.pattern, pattern, best_parent.pattern):
                            
                            combined = self._create_subsumption_pattern([sibling.pattern, pattern])
                            
                            # Garder le pattern le plus spécifique
                            if best_combined_pattern is None or \
                               (combined[0].count('*') + combined[1].count('*') < 
                                best_combined_pattern[0].count('*') + best_combined_pattern[1].count('*')):
                                best_sibling = sibling
                                best_combined_pattern = combined
                    
                    if best_sibling and best_combined_pattern:
                        # Créer le nœud intermédiaire
                        intermediate = self._get_or_create_node(best_combined_pattern, False)
                        
                        # Gérer les relations existantes
                        if intermediate.parent and intermediate.parent != best_parent:
                            intermediate.parent.children.remove(intermediate)
                        
                        # Établir les nouvelles relations
                        if intermediate not in best_parent.children:
                            best_parent.children.append(intermediate)
                        intermediate.parent = best_parent
                        
                        # Déplacer le sibling
                        if best_sibling.parent == best_parent:
                            best_parent.children.remove(best_sibling)
                        best_sibling.parent = intermediate
                        if best_sibling not in intermediate.children:
                            intermediate.children.append(best_sibling)
                        
                        # Ajouter la nouvelle feuille
                        new_leaf.parent = intermediate
                        intermediate.children.append(new_leaf)
                        
                        # Réorganiser
                        self._reorganize_tree(intermediate, best_parent)
                        
                        return
            
            # Ajouter directement comme enfant
            new_leaf.parent = best_parent
            best_parent.children.append(new_leaf)
            
        else:
            # Aucun subsumeur trouvé
            new_root_pattern = self._create_subsumption_pattern([self.root.pattern, pattern])
            new_root = self._get_or_create_node(new_root_pattern, False)
            
            self.root.parent = new_root
            new_leaf.parent = new_root
            new_root.children = [self.root, new_leaf]
            self.root = new_root
    
    def _verify_tree_structure(self) -> bool:
        """
        Vérifie l'intégrité de l'arbre.
        """
        issues = []
        
        # Vérifier l'unicité des patterns non-feuilles
        pattern_count = {}
        for node in self.all_nodes:
            if not node.is_leaf:
                pattern = node.pattern
                if pattern not in pattern_count:
                    pattern_count[pattern] = []
                pattern_count[pattern].append(node.id)
        
        for pattern, node_ids in pattern_count.items():
            if len(node_ids) > 1:
                issues.append(f"Pattern {pattern} dupliqué: nœuds {node_ids}")
        
        # Vérifier que chaque nœud n'a qu'un parent
        parent_count = {}
        
        def count_parents(node: Node):
            for child in node.children:
                parent_count[child.id] = parent_count.get(child.id, 0) + 1
                count_parents(child)
        
        if self.root:
            count_parents(self.root)
        
        for node_id, count in parent_count.items():
            if count > 1:
                issues.append(f"Nœud {node_id} a {count} parents")
        
        if issues:
            print("Problèmes détectés:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        
        return True
    
    def _calculate_positions(self) -> Dict[int, Tuple[float, float]]:
        """
        Calcule les positions des nœuds pour la visualisation.
        """
        positions = {}
        levels = {}
        
        # Première passe : déterminer les niveaux
        def assign_levels(node: Node, level: int = 0):
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
        def calculate_subtree_width(node: Node) -> int:
            if node.is_leaf:
                return 1
            return max(1, sum(calculate_subtree_width(child) for child in node.children))
        
        def assign_positions_improved(node: Node, x: float = 0, y: float = 0, 
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
    
    def visualize(self, figsize: Tuple[int, int] = (20, 12)):
        """
        Visualise l'arbre hiérarchique.
        """
        if self.root is None:
            print("L'arbre est vide.")
            return
        
        # Vérifier la structure
        print("Vérification de la structure de l'arbre...")
        if not self._verify_tree_structure():
            print("⚠️  Des problèmes ont été détectés dans la structure!")
        else:
            print("✅ Structure correcte - pas de duplication de patterns")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculer les positions
        pos = self._calculate_positions()
        
        # Créer le graphe pour les arêtes
        G = nx.DiGraph()
        
        # MODIFICATION ICI : Vérifier que les nœuds existent avant d'ajouter les arêtes
        valid_node_ids = {node.id for node in self.all_nodes}
        
        def add_edges(node: Node):
            if node.id not in valid_node_ids:
                return
            for child in node.children:
                # Vérifier que l'enfant existe aussi dans all_nodes
                if child.id in valid_node_ids:
                    G.add_edge(node.id, child.id)
                    add_edges(child)
        
        if self.root and self.root.id in valid_node_ids:
            add_edges(self.root)
        
        # Dessiner les arêtes seulement si elles existent
        if G.edges():
            # Filtrer les positions pour ne garder que les nœuds valides
            valid_pos = {node_id: position for node_id, position in pos.items() 
                        if node_id in valid_node_ids}
            
            nx.draw_networkx_edges(G, valid_pos, edge_color='gray', arrows=True, 
                                arrowsize=15, ax=ax, width=2, alpha=0.6)
        
        # Dessiner les nœuds avec leurs labels (reste identique)
        for node in self.all_nodes:
            if node.id in pos:
                x, y = pos[node.id]
                
                # Label
                label = f"{node.pattern[0]}\n{node.pattern[1]}"
                if node.is_leaf and node.original_data:
                    label += f"\n{node.original_data}"
                
                # Couleur et style
                if node.is_leaf:
                    color = 'lightgreen'
                    fontsize = 9
                    weight = 'bold'
                else:
                    color = 'lightblue'
                    fontsize = 10
                    weight = 'normal'
                
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
        y_margin = (max(all_y) - min(all_y)) * 0.1
        
        ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
        ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
        
        ax.set_title("Arbre Hiérarchique 2D - Sans Duplication", fontsize=16)
        ax.axis('off')
        
        # Statistiques
        leaf_count = sum(1 for n in self.all_nodes if n.is_leaf)
        proto_count = sum(1 for n in self.all_nodes if not n.is_leaf)
        unique_patterns = len(self.pattern_to_node)
        
        # Légende avec statistiques
        leaf_patch = mpatches.Patch(color='lightgreen', label=f'Données/Feuilles ({leaf_count})')
        node_patch = mpatches.Patch(color='lightblue', label=f'Prototypes/Nœuds ({proto_count})')
        info_text = f'Patterns uniques: {unique_patterns}'
        
        ax.legend(handles=[leaf_patch, node_patch], 
                title=info_text,
                loc='upper right', fontsize=12)
        
        plt.tight_layout()
        plt.show()
    
    def print_tree(self, node: Optional[Node] = None, depth: int = 0):
        """
        Affiche l'arbre sous forme textuelle.
        """
        if node is None:
            if self.root is None:
                print("L'arbre est vide.")
                return
            node = self.root
        
        indent = "  " * depth
        pattern_str = f"({node.pattern[0]}, {node.pattern[1]})"
        
        if node.is_leaf:
            print(f"{indent}├─ {pattern_str} [Donnée: {node.original_data}]")
        else:
            # Afficher l'ID pour les prototypes pour vérifier l'unicité
            print(f"{indent}├─ {pattern_str} [Prototype ID:{node.id} - {len(node.children)} enfants]")
        
        for child in node.children:
            self.print_tree(child, depth + 1)
        
        
    def calculate_mdl_scores(self):
        """
        Calcule les scores MDL pour chaque prototype dans l'arbre.
        
        Score = rang (profondeur) + popularité
        Popularité = nombre de fils directs qui sont des données (valeur négative)
        
        Returns:
            Dict[int, float]: Dictionnaire {node_id: score}
        """
        scores = {}
        
        def calculate_scores_recursive(node: Node, depth: int = 0):
            if node.is_leaf:
                # Les feuilles n'ont pas de score MDL
                return
            
            # Calcul du rang (profondeur)
            rank = depth
            
            # Calcul de la popularité (nombre de fils données)
            popularity = 0
            for child in node.children:
                if child.is_leaf:
                    popularity -= 1  # -1 pour chaque donnée fille
            
            # Score = rang + popularité
            if node == self.root:
                # La racine a un coût infini
                scores[node.id] = float('inf')
            else:
                scores[node.id] = rank + popularity
            
            # Récursion sur les enfants
            for child in node.children:
                calculate_scores_recursive(child, depth + 1)
        
        if self.root:
            calculate_scores_recursive(self.root)
        
        return scores
    

    
    def apply_mdl_pruning(self, verbose: bool = False):
        """
        Applique l'algorithme MDL pour élaguer l'arbre.
        Garde les prototypes avec les scores minimaux et rattache les données
        aux prototypes gardés.
        
        Args:
            verbose: Si True, affiche des informations sur le processus
            
        Returns:
            List[Node]: Liste des prototypes gardés
        """
        if not self.root:
            return []
        
        # Calculer les scores MDL
        scores = self.calculate_mdl_scores()
        
        # Identifier les prototypes à garder
        kept_prototypes = set()
        kept_prototypes.add(self.root.id)  # Toujours garder la racine
        
        def has_data_children(node: Node) -> bool:
            """Vérifie si un nœud a des enfants données."""
            return any(child.is_leaf for child in node.children)
        
        def get_min_child_score(node: Node) -> float:
            """Retourne le score minimum parmi les enfants prototypes."""
            child_scores = []
            for child in node.children:
                if not child.is_leaf:
                    child_scores.append(scores.get(child.id, float('inf')))
            return min(child_scores) if child_scores else float('inf')
        
        def should_keep_prototype(node: Node, parent_score: float) -> bool:
            """
            Détermine si un prototype doit être gardé.
            
            Un prototype est supprimé si :
            - Il n'a pas de données comme fils directs (popularité = 0)
            - Au moins un de ses enfants prototypes a un meilleur score
            """
            if node.is_leaf:
                return False
            
            node_score = scores.get(node.id, float('inf'))
            
            # Si le nœud n'a pas de données directes
            if not has_data_children(node):
                min_child_score = get_min_child_score(node)
                # Si un enfant a un meilleur score, ne pas garder ce nœud
                if min_child_score < node_score:
                    if verbose:
                        print(f"Prototype {node.pattern} supprimé: pas de données directes et enfant avec meilleur score ({min_child_score} < {node_score})")
                    return False
            
            # Règles originales
            children_scores = []
            for child in node.children:
                if not child.is_leaf:
                    children_scores.append(scores.get(child.id, float('inf')))
            
            # Si le nœud a un meilleur score que tous ses enfants prototypes
            if children_scores and all(node_score <= child_score for child_score in children_scores):
                return True
            
            return node_score < parent_score
        
        # Parcourir l'arbre pour identifier les prototypes à garder
        def identify_kept_prototypes(node: Node, parent_score: float = float('inf')):
            if node.is_leaf:
                return
            
            node_score = scores.get(node.id, float('inf'))
            
            # Cas spécial : nœud sans données directes
            if not has_data_children(node):
                min_child_score = get_min_child_score(node)
                if min_child_score < node_score:
                    # Ne pas garder ce nœud, propager le score du parent
                    if verbose:
                        print(f"  Saut du prototype {node.pattern} (score {node_score}), propagation aux enfants")
                    for child in node.children:
                        identify_kept_prototypes(child, parent_score)
                    return
            
            if should_keep_prototype(node, parent_score):
                kept_prototypes.add(node.id)
                if verbose:
                    print(f"  Prototype gardé: {node.pattern} (score {node_score})")
                # Continuer avec ce score comme référence
                for child in node.children:
                    identify_kept_prototypes(child, node_score)
            else:
                # Ne pas garder ce prototype, continuer avec le score parent
                if verbose:
                    print(f"  Prototype non gardé: {node.pattern} (score {node_score})")
                for child in node.children:
                    identify_kept_prototypes(child, parent_score)
        
        if verbose:
            print("\nProcessus d'élagage MDL:")
        
        identify_kept_prototypes(self.root)
        
        # Créer le nouvel arbre élagué
        self._reorganize_after_mdl(kept_prototypes)
        
        return [node for node in self.all_nodes if node.id in kept_prototypes]
    
    def _reorganize_after_mdl(self, kept_prototypes: Set[int]):
        """
        Réorganise l'arbre après l'élagage MDL.
        Rattache les données aux prototypes gardés les plus proches.
        """
        # Créer une correspondance des nœuds à leur nouveau parent
        new_parent_map = {}
        
        def find_nearest_kept_ancestor(node: Node) -> Optional[Node]:
            """Trouve l'ancêtre gardé le plus proche."""
            current = node.parent
            while current:
                if current.id in kept_prototypes:
                    return current
                current = current.parent
            return None
        
        # Parcourir tous les nœuds pour établir les nouvelles relations
        nodes_to_update = []
        for node in self.all_nodes:
            if node.is_leaf or (not node.is_leaf and node.id not in kept_prototypes):
                # C'est une donnée ou un prototype non gardé
                if node.parent and node.parent.id not in kept_prototypes:
                    # Son parent direct n'est pas gardé, trouver le nouveau parent
                    new_parent = find_nearest_kept_ancestor(node)
                    if new_parent:
                        nodes_to_update.append((node, new_parent))
        
        # Appliquer les mises à jour
        for node, new_parent in nodes_to_update:
            # Retirer du parent actuel
            if node.parent:
                node.parent.children.remove(node)
            
            # Ajouter au nouveau parent
            node.parent = new_parent
            if node not in new_parent.children:
                new_parent.children.append(node)
        
        # Supprimer les prototypes non gardés de la structure
        prototypes_to_remove = [node for node in self.all_nodes 
                               if not node.is_leaf and node.id not in kept_prototypes]
        
        for proto in prototypes_to_remove:
            self.all_nodes.remove(proto)
            if proto.pattern in self.pattern_to_node:
                del self.pattern_to_node[proto.pattern]
    
    def visualize_with_mdl_scores(self, figsize: Tuple[int, int] = (20, 12), 
                                 show_scores: bool = True):
        """
        Visualise l'arbre avec les scores MDL affichés.
        
        Args:
            figsize: Taille de la figure
            show_scores: Si True, affiche les scores MDL sur les nœuds
        """
        if self.root is None:
            print("L'arbre est vide.")
            return
        
        # Calculer les scores MDL
        scores = self.calculate_mdl_scores()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculer les positions
        pos = self._calculate_positions()
        
        # Créer le graphe pour les arêtes
        G = nx.DiGraph()
        
        def add_edges(node: Node):
            for child in node.children:
                G.add_edge(node.id, child.id)
                add_edges(child)
        
        if self.root:
            add_edges(self.root)
        
        # Dessiner les arêtes
        if G.edges():
            nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                                 arrowsize=15, ax=ax, width=2, alpha=0.6)
        
        # Dessiner les nœuds avec leurs labels et scores
        for node in self.all_nodes:
            if node.id in pos:
                x, y = pos[node.id]
                
                # Label de base
                label = f"{node.pattern[0]}\n{node.pattern[1]}"
                if node.is_leaf and node.original_data:
                    label += f"\n{node.original_data}"
                
                # Ajouter le score MDL pour les prototypes
                if show_scores and not node.is_leaf:
                    score = scores.get(node.id, float('inf'))
                    if score == float('inf'):
                        score_text = "SCORE : ∞"
                    else:
                        score_text = f"SCORE : {int(score)}"
                    
                    # Afficher le score au-dessus du nœud
                    ax.text(x, y + 0.5, score_text, ha='center', va='bottom',
                           fontsize=10, weight='bold')
                
                # Couleur et style
                if node.is_leaf:
                    color = 'lightgreen'
                    fontsize = 9
                    weight = 'bold'
                else:
                    color = 'lightblue'
                    fontsize = 10
                    weight = 'normal'
                
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
        y_margin = (max(all_y) - min(all_y)) * 0.1 + 1  # Plus de marge pour les scores
        
        ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
        ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
        
        ax.set_title("Arbre Hiérarchique avec Scores MDL", fontsize=16)
        ax.axis('off')
        
        # Légende
        leaf_patch = mpatches.Patch(color='lightgreen', label='Données (feuilles)')
        node_patch = mpatches.Patch(color='lightblue', label='Prototypes (nœuds)')
        ax.legend(handles=[leaf_patch, node_patch], loc='upper right', fontsize=12)
        
        plt.tight_layout()
        plt.show()
        
    def get_cluster_assignments(self):
        """
        Assigne chaque donnée à son prototype parent.
        La racine n'est pas considérée comme un cluster.
        Les données directement attachées à la racine sont des points isolés.
        
        Returns:
            Dict[int, Tuple[Node, List[Tuple[int, int]]]]: Dictionnaire {node_id: (node, [données])}
        """
        clusters = {}
        
        # Pour chaque nœud prototype (sauf la racine)
        for node in self.all_nodes:
            if not node.is_leaf and node != self.root:
                # Collecter les données directement sous ce prototype
                data_points = []
                for child in node.children:
                    if child.is_leaf:
                        data_points.append(child.original_data)
                
                if data_points:  # Si le prototype a des données directes
                    clusters[node.id] = (node, data_points)
        
        # Traiter les données isolées (directement attachées à la racine)
        if self.root:
            isolated_counter = -1  # IDs négatifs pour les points isolés
            for child in self.root.children:
                if child.is_leaf:
                    # Chaque donnée isolée forme son propre cluster
                    clusters[isolated_counter] = (None, [child.original_data])
                    isolated_counter -= 1
        
        return clusters

    def calculate_prototype_centers(self):
        """
        Calcule le centre (médiane) de chaque prototype basé sur ses données.
        Pour les points isolés, le centre est le point lui-même.
        
        Returns:
            Dict[int, Tuple[float, float]]: Dictionnaire {node_id: (x_median, y_median)}
        """
        clusters = self.get_cluster_assignments()
        centers = {}
        
        for node_id, (node, data_points) in clusters.items():
            if data_points:
                # Si c'est un point isolé (node est None)
                if node is None:
                    # Le centre est le point lui-même
                    centers[node_id] = (float(data_points[0][0]), float(data_points[0][1]))
                else:
                    # Calculer la médiane pour un vrai cluster
                    x_coords = [point[0] for point in data_points]
                    y_coords = [point[1] for point in data_points]
                    
                    x_median = np.median(x_coords)
                    y_median = np.median(y_coords)
                    
                    centers[node_id] = (x_median, y_median)
        
        return centers

    def visualize_clusters_2d(self, figsize: Tuple[int, int] = (12, 8), 
                            show_centers: bool = True,
                            show_legend: bool = True,
                            isolated_marker: str = 'D'):
        """
        Visualise les clusters en 2D avec des couleurs différentes.
        Les points isolés sont affichés avec un marqueur différent.
        
        Args:
            figsize: Taille de la figure
            show_centers: Si True, affiche les centres (médianes) des clusters
            show_legend: Si True, affiche la légende
            isolated_marker: Marqueur pour les points isolés
        """
        clusters = self.get_cluster_assignments()
        centers = self.calculate_prototype_centers()
        
        if not clusters:
            print("Aucun cluster à visualiser.")
            return
        
        # Créer la figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Séparer les clusters normaux et les points isolés
        normal_clusters = {k: v for k, v in clusters.items() if v[0] is not None}
        isolated_points = {k: v for k, v in clusters.items() if v[0] is None}
        
        # Générer des couleurs distinctes
        total_clusters = len(clusters)
        colors = plt.cm.tab10(np.linspace(0, 1, total_clusters))
        if total_clusters > 10:
            colors = plt.cm.tab20(np.linspace(0, 1, total_clusters))
        
        color_idx = 0
        
        # Tracer les clusters normaux
        for node_id, (node, data_points) in normal_clusters.items():
            if data_points:
                # Séparer les coordonnées
                x_coords = [point[0] for point in data_points]
                y_coords = [point[1] for point in data_points]
                
                # Tracer les points
                scatter = ax.scatter(x_coords, y_coords, 
                                c=[colors[color_idx]], 
                                label=f'Cluster {node.pattern}',
                                alpha=0.7, 
                                s=100,
                                edgecolors='black',
                                linewidth=0.5)
                
                # Tracer le centre si demandé
                if show_centers and node_id in centers:
                    center_x, center_y = centers[node_id]
                    ax.scatter(center_x, center_y, 
                            c=[colors[color_idx]], 
                            marker='X', 
                            s=300, 
                            edgecolors='black',
                            linewidth=2)
                    
                    # Ajouter une annotation pour le centre
                    ax.annotate(f'{node.pattern}', 
                            xy=(center_x, center_y),
                            xytext=(5, 5), 
                            textcoords='offset points',
                            fontsize=8,
                            bbox=dict(boxstyle='round,pad=0.3', 
                                    facecolor=colors[color_idx], 
                                    alpha=0.3))
                
                color_idx += 1
        
        # Tracer les points isolés
        for node_id, (_, data_points) in isolated_points.items():
            if data_points:
                x = data_points[0][0]
                y = data_points[0][1]
                
                # Tracer le point isolé avec un marqueur différent
                ax.scatter(x, y, 
                        c=[colors[color_idx]], 
                        marker=isolated_marker,
                        label=f'Isolé ({x}, {y})',
                        alpha=0.8, 
                        s=150,
                        edgecolors='black',
                        linewidth=1.5)
                
                # Annotation pour le point isolé
                ax.annotate(f'Isolé\n({x}, {y})', 
                        xy=(x, y),
                        xytext=(10, 10), 
                        textcoords='offset points',
                        fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='yellow', 
                                alpha=0.5))
                
                color_idx += 1
        
        # Configuration du graphique
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_title('Visualisation 2D des Clusters (sans la racine)', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Ajuster les limites des axes
        all_data = [point for (_, points) in clusters.values() for point in points]
        if all_data:
            x_coords = [p[0] for p in all_data]
            y_coords = [p[1] for p in all_data]
            
            x_margin = (max(x_coords) - min(x_coords)) * 0.1
            y_margin = (max(y_coords) - min(y_coords)) * 0.1
            
            ax.set_xlim(min(x_coords) - x_margin, max(x_coords) + x_margin)
            ax.set_ylim(min(y_coords) - y_margin, max(y_coords) + y_margin)
        
        # Légende
        if show_legend and total_clusters <= 15:  # Limiter la légende si trop de clusters
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
        
        # Afficher les statistiques
        print(f"\nStatistiques des clusters:")
        print(f"Nombre total de clusters: {len(clusters)}")
        print(f"  - Clusters normaux: {len(normal_clusters)}")
        print(f"  - Points isolés: {len(isolated_points)}")
        
        for node_id, (node, data_points) in normal_clusters.items():
            print(f"\nCluster {node.pattern}: {len(data_points)} points")
            if node_id in centers:
                print(f"  Centre (médiane): {centers[node_id]}")
        
        if isolated_points:
            print(f"\nPoints isolés:")
            for node_id, (_, data_points) in isolated_points.items():
                print(f"  {data_points[0]}")

   

def test_with_6_points():
    """
    Teste avec les 6 points fournis.
    """
    print("=== Test avec 6 points (sans duplication) ===\n")
    
    tree = BinaryHierarchicalTree(n_bits=4)
    
    data_points = [
        (0b1110, 0b1010),  # (14, 10)
        (0b1101, 0b1011),  # (13, 11)
        (0b1011, 0b1100),  # (11, 12)
        (0b1111, 0b1101),  # (15, 13)
        (0b1100, 0b1110),  # (12, 14)
        (0b1010, 0b1111),  # (10, 15)
        (0b1001, 0b1001),   # (9, 9)
        (0b1000, 0b1000)   # (8, 8)
    ]
    
    for i, data in enumerate(data_points):
        print(f"\nInsertion {i+1}: {data} = ({data[0]:04b}, {data[1]:04b})")
        tree.insert(data)
        
    scores = tree.calculate_mdl_scores()
    
    tree.visualize_with_mdl_scores()
    
    kept_prototypes = tree.apply_mdl_pruning()

    
    print("\n=== Structure finale ===")
    tree.print_tree()
    
    print(f"\nNombre de patterns uniques (prototypes): {len(tree.pattern_to_node)}")
    print("Patterns existants:")
    for pattern in sorted(tree.pattern_to_node.keys()):
        print(f"  {pattern}")
    
    print("\n=== Visualisation ===")
    tree.visualize()
    

    
    return tree


def test_with_random_data(n_points: int = 20, n_bits: int = 4, seed: int = 42):
    """
    Teste l'arbre avec des données aléatoires.
    """
    print(f"\n=== Test avec {n_points} points aléatoires (sans duplication) ===\n")
    
    np.random.seed(seed)
    
    tree = BinaryHierarchicalTree(n_bits=n_bits)
    
    # Générer des données
    max_val = 2**n_bits - 1
    data_points = []
    
    # Clusters
    cluster_centers = [(7, 8), (12, 3), (3, 14), (14, 12)]
    
    for i, center in enumerate(cluster_centers):
        n_cluster_points = n_points // len(cluster_centers)
        for j in range(n_cluster_points):
            noise_x = np.random.randint(-2, 3)
            noise_y = np.random.randint(-2, 3)
            
            x = max(0, min(max_val, center[0] + noise_x))
            y = max(0, min(max_val, center[1] + noise_y))
            
            data_points.append((x, y))
    
    # Points isolés
    remaining = n_points - len(data_points)
    for i in range(remaining):
        x = np.random.randint(0, max_val + 1)
        y = np.random.randint(0, max_val + 1)
        data_points.append((x, y))
    
    # Insérer les données
    for i, data in enumerate(data_points):
        tree.insert(data)
    
    print(f"\nPatterns uniques créés: {len(tree.pattern_to_node)}")
    print("Liste des patterns:")
    for pattern in sorted(tree.pattern_to_node.keys(), 
                         key=lambda p: (p[0].count('*'), p[1].count('*'))):
        node = tree.pattern_to_node[pattern]
        print(f"  {pattern} - {len(node.children)} enfants")
    
    scores = tree.calculate_mdl_scores()
    
    tree.visualize_with_mdl_scores()
    
    kept_prototypes = tree.apply_mdl_pruning()
    
    
    print("\n=== Structure finale ===")
    print("Nombre de prototypes gardés:", len(kept_prototypes))

    
    tree.visualize()
    
    tree.visualize_clusters_2d(show_centers=True)
    
    # Visualiser les frontières
    print("\n=== Visualisation des frontières ===")
    tree.visualize_cluster_boundaries(resolution=50)
    
    return tree, data_points

def test_with_random_data_gaussienne(n_points: int = 20, n_bits: int = 8, seed: int = 42):
    """
    Teste l'arbre avec des données aléatoires.
    """
    print(f"\n=== Test avec {n_points} points aléatoires (sans duplication) ===\n")
    
    np.random.seed(seed)
    
    tree = BinaryHierarchicalTree(n_bits=n_bits)
    
    # Générer des données
    max_val = 2**n_bits - 1
    data_points = []
    
    # Clusters
    cluster_centers = [(150, 50), (1000, 1000), (50, 50), (40, 20)]
    
    for i, center in enumerate(cluster_centers):
        n_cluster_points = n_points // len(cluster_centers)
        for j in range(n_cluster_points):
            noise_x = np.random.randint(-5, 5)
            noise_y = np.random.randint(-5, 5)
            
            x = max(0, min(max_val, center[0] + noise_x))
            y = max(0, min(max_val, center[1] + noise_y))
            
            data_points.append((x, y))
    
    # Points isolés
    remaining = n_points - len(data_points)
    for i in range(remaining):
        x = np.random.randint(0, max_val + 1)
        y = np.random.randint(0, max_val + 1)
        data_points.append((x, y))
    
    # Insérer les données
    for i, data in enumerate(data_points):
        tree.insert(data)
    
    print(f"\nPatterns uniques créés: {len(tree.pattern_to_node)}")
    print("Liste des patterns:")
    for pattern in sorted(tree.pattern_to_node.keys(), 
                         key=lambda p: (p[0].count('*'), p[1].count('*'))):
        node = tree.pattern_to_node[pattern]
        print(f"  {pattern} - {len(node.children)} enfants")
    
    scores = tree.calculate_mdl_scores()
    
    tree.visualize_with_mdl_scores()
    
    kept_prototypes = tree.apply_mdl_pruning()
    
    
    print("\n=== Structure finale ===")
    print("Nombre de prototypes gardés:", len(kept_prototypes))

    
    tree.visualize()
    
    tree.visualize_clusters_2d(show_centers=True)
    
    # Visualiser les frontières
    
    return tree, data_points


def test_duplicate_data_issue():
    """
    Teste spécifiquement le cas des données dupliquées.
    """
    print("=== Test du problème des données dupliquées ===\n")
    
    tree = BinaryHierarchicalTree(n_bits=4)
    
    # Cas problématique : deux données identiques
    print("Insertion de données identiques (1110, 1110):")
    tree.insert((0b1110, 0b1110))  # (14, 14)
    tree.insert((0b1110, 0b1110))  # (14, 14) - identique
    
    print("\nStructure après 2 insertions identiques:")
    tree.print_tree()
    
    # Ajouter une donnée légèrement différente
    print("\nAjout d'une donnée proche mais différente (1110, 1111):")
    tree.insert((0b1110, 0b1111))  # (14, 15)
    
    print("\nStructure après 3 insertions:")
    tree.print_tree()
    
    # Ajouter plus de données pour voir l'évolution
    print("\nAjout de données supplémentaires:")
    tree.insert((0b1100, 0b1100))  # (12, 12)
    tree.insert((0b1101, 0b1101))  # (13, 13)
    
    print("\nStructure avant simplification:")
    tree.print_tree()
    
    # Simplifier l'arbre
    print("\n--- Simplification de l'arbre ---")
    tree.simplify_tree()
    
    print("\nStructure après simplification:")
    tree.print_tree()
    
    print(f"\nPatterns uniques créés: {len(tree.pattern_to_node)}")
    print("Patterns existants:")
    for pattern in sorted(tree.pattern_to_node.keys(), 
                         key=lambda p: (p[0].count('*'), p[1].count('*'))):
        print(f"  {pattern}")
    
    tree.visualize(figsize=(16, 10))
    
    # Visualiser les clusters
    print("\n=== Visualisation des clusters ===")
    tree.visualize_clusters_2d(show_centers=True)
    
    # Visualiser les frontières
    print("\n=== Visualisation des frontières ===")
    tree.visualize_cluster_boundaries(resolution=50)
    
    return tree

def test_cluster_visualization():
    """
    Test de la visualisation des clusters.
    """
    print("=== Test de visualisation des clusters ===\n")
    
    # Créer un arbre et insérer des données
    tree = BinaryHierarchicalTree(n_bits=4)
    
    # Créer des clusters distincts
    cluster_data = [
        # Cluster 1: coin bas-gauche
        [(2, 2), (3, 2), (2, 3), (3, 3), (4, 3)],
        # Cluster 2: coin haut-droit
        [(12, 12), (13, 12), (12, 13), (13, 13), (14, 13)],
        # Cluster 3: milieu
        [(7, 7), (8, 7), (7, 8), (8, 8), (9, 8)],
        # Données isolées
        [(1, 14), (14, 1), (5, 11)]
    ]
    
    # Insérer toutes les données
    for cluster in cluster_data:
        for point in cluster:
            tree.insert(point)
    
    print("Structure de l'arbre:")
    tree.print_tree()
    
    # Appliquer MDL
    print("\nApplication de l'élagage MDL...")
    tree.apply_mdl_pruning(verbose=True)
    
    print("\nStructure après MDL:")
    tree.print_tree()
    
    # Visualiser les clusters
    print("\n=== Visualisation des clusters ===")
    tree.visualize_clusters_2d(show_centers=True)
    
    # Visualiser les frontières
    print("\n=== Visualisation des frontières ===")
    tree.visualize_cluster_boundaries(resolution=50)
    
    return tree

    
if __name__ == "__main__":



    # Test avec les 6 points
    # tree1 = test_with_6_points()
    
    # Test avec données aléatoires
    # print("\n" + "="*70 + "\n")
    tree = test_with_random_data_gaussienne()
    tree2, data = test_with_random_data(n_points=20)