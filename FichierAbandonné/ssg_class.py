import math
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict

class SGGGraph:
    def __init__(self, numbers):
        self.numbers = numbers
        self.max_number = max(numbers)
        self.n_bits = math.ceil(math.log2(self.max_number + 1))
        self.levels = dict()
        self.edges = []
        self.graph = nx.DiGraph()

    def _to_binary_strings(self):
        return [format(n, f'0{self.n_bits}b') for n in self.numbers]

    def generate_levels(self):
        level0 = self._to_binary_strings()
        self.levels[0] = set(level0)

        # Première passe : création des SGG à partir des binaires
        for L in range(1, self.n_bits + 1):
            grouping = defaultdict(set)
            for b in level0:
                key = b[:self.n_bits - L]
                candidate = key + "*" * L
                grouping[candidate].add(b)
            for candidate, group in grouping.items():
                if len(group) >= 2:
                    self.levels.setdefault(L, set()).add(candidate)
                    for child in group:
                        self.edges.append((child, candidate))

        # Seconde passe : généralisation entre niveaux
        for L in range(1, self.n_bits):
            if L not in self.levels:
                continue
            grouping = defaultdict(set)
            for node in self.levels[L]:
                fixed = node.split("*")[0]
                key = fixed[:-1] if fixed else ""
                candidate = key + "*" * (L + 1)
                grouping[candidate].add(node)
            for candidate, group in grouping.items():
                if len(group) >= 2:
                    self.levels.setdefault(L + 1, set()).add(candidate)
                    for child in group:
                        self.edges.append((child, candidate))

    def build_graph(self):
        for lvl, nodes in self.levels.items():
            for node in nodes:
                self.graph.add_node(node, level=lvl)
        for src, dst in self.edges:
            self.graph.add_edge(src, dst)

    def draw(self):
        # Positionnement vertical par niveau
        pos = {}
        for lvl, nodes in self.levels.items():
            for i, node in enumerate(sorted(nodes)):
                pos[node] = (i, -lvl)

        # Tracé du graphe
        plt.figure(figsize=(18, 10))
        nx.draw(
            self.graph, pos, with_labels=True,
            node_size=1800, node_color='lightblue',
            font_size=10, arrows=True
        )
        plt.title("Hiérarchie des SGG généralisable pour toute liste de nombres")
        plt.axis('off')
        plt.show()

    def run(self):
        self.generate_levels()
        self.build_graph()
        self.draw()

numbers = [36, 37, 40, 45, 51, 52, 55, 60, 62, 64]

sgg = SGGGraph(numbers)
sgg.run()
