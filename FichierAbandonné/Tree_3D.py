import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

@dataclass
class Node:
    pattern: Tuple[str, str, str]
    children: List['Node'] = field(default_factory=list)
    parent: Optional['Node'] = None
    is_leaf: bool = False
    original_data: Optional[Tuple[int, int, int]] = None
    id: int = 0

class BinaryHierarchicalTree3D:
    def __init__(self, n_bits: int = 4):
        self.n_bits = n_bits
        self.root = None
        self.node_counter = 0
        self.all_nodes = []
        self.pattern_to_node = {}

    def _to_binary(self, value: int) -> str:
        return format(value, f'0{self.n_bits}b')

    def _can_subsume(self, pattern: Tuple[str, str, str], data_pattern: Tuple[str, str, str]) -> bool:
        return all(p == '*' or p == d for p_comp, d_comp in zip(pattern, data_pattern) for p, d in zip(p_comp, d_comp))

    def _create_subsumption_pattern(self, patterns: List[Tuple[str, str, str]]) -> Tuple[str, str, str]:
        def merge(index):
            res = ''
            for i in range(self.n_bits):
                bits = [p[index][i] for p in patterns]
                if all(b == bits[0] for b in bits):
                    res += bits[0]
                else:
                    res += '*' * (self.n_bits - i)
                    break
            return res
        return merge(0), merge(1), merge(2)

    def _get_or_create_node(self, pattern, is_leaf, original_data=None):
        if is_leaf:
            node = Node(pattern, is_leaf=True, original_data=original_data, id=self.node_counter)
            self.node_counter += 1
            self.all_nodes.append(node)
            return node
        if pattern in self.pattern_to_node:
            return self.pattern_to_node[pattern]
        node = Node(pattern, is_leaf=False, id=self.node_counter)
        self.node_counter += 1
        self.all_nodes.append(node)
        self.pattern_to_node[pattern] = node
        return node

    def insert(self, data: Tuple[int, int, int]):
        pattern = tuple(self._to_binary(d) for d in data)
        if self.root is None:
            self.root = self._get_or_create_node(pattern, True, data)
        else:
            self._insert_into_tree(pattern, data)

    def _find_all_subsumers(self, pattern: Tuple[str, str, str]) -> List[Node]:
        subsumers = []
        def dfs(node):
            if not node.is_leaf and self._can_subsume(node.pattern, pattern):
                subsumers.append(node)
            for c in node.children:
                dfs(c)
        if self.root:
            dfs(self.root)
        return sorted(subsumers, key=lambda n: sum(comp.count('*') for comp in n.pattern))

    def _insert_into_tree(self, pattern, data):
        new_leaf = self._get_or_create_node(pattern, True, data)
        subsumers = self._find_all_subsumers(pattern)
        if subsumers:
            best_parent = subsumers[0]
            new_leaf.parent = best_parent
            best_parent.children.append(new_leaf)
        else:
            new_root_pattern = self._create_subsumption_pattern([self.root.pattern, pattern])
            new_root = self._get_or_create_node(new_root_pattern, False)
            self.root.parent = new_root
            new_leaf.parent = new_root
            new_root.children = [self.root, new_leaf]
            self.root = new_root

    def _calculate_positions(self):
        positions = {}
        def dfs(node, x=0, y=0, z=0):
            positions[node.id] = (x, y, z)
            for i, c in enumerate(node.children):
                dfs(c, x + i - len(node.children) / 2, y - 2, z - 2)
        if self.root:
            dfs(self.root)
        return positions

    def visualize(self):
        if self.root is None:
            print("L'arbre est vide.")
            return

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        pos = self._calculate_positions()

        for node in self.all_nodes:
            x, y, z = pos[node.id]
            ax.text(x, y, z, f"{node.pattern[0]}\n{node.pattern[1]}\n{node.pattern[2]}",
                    bbox=dict(facecolor='lightgreen' if node.is_leaf else 'lightblue', alpha=0.6),
                    ha='center', va='center')
            if node.parent:
                x2, y2, z2 = pos[node.parent.id]
                ax.plot([x, x2], [y, y2], [z, z2], color='gray')

        ax.set_title("Arbre Hiérarchique 3D")
        ax.set_axis_off()
        plt.show()

    def print_tree(self, node=None, depth=0):
        if node is None:
            if self.root is None:
                print("L'arbre est vide.")
                return
            node = self.root
        indent = "  " * depth
        if node.is_leaf:
            print(f"{indent}- {node.pattern} [Donnée: {node.original_data}]")
        else:
            print(f"{indent}- {node.pattern} [Prototype ID:{node.id}]")
        for c in node.children:
            self.print_tree(c, depth + 1)


def test_tree_3d():
    tree = BinaryHierarchicalTree3D(n_bits=4)
    data_points = [
        (14, 10, 1),
        (13, 11, 2),
        (11, 12, 3),
        (15, 13, 4),
        (12, 14, 5),
        (10, 15, 6),
    ]
    for point in data_points:
        tree.insert(point)
    tree.print_tree()
    tree.visualize()

test_tree_3d()
