"""
Telecom Paris : projet Contraste

Projet réalisé par : Samy Bali, Hussein El gouch, Issa Daou, Anas El Barhdadi

Encadré par Jean-Louis Dessalles
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
from networkx.drawing.nx_pydot import graphviz_layout
from collections import defaultdict

# ======================
# Utils bit-à-bit, conversion, préfixe commun
# ======================
class BitUtils:
    @staticmethod
    def bin_sub(a: str, b: str) -> str:
        L, res, borrow = len(a), ['0']*len(a), 0
        for i in range(L-1, -1, -1):
            ai, bi = int(a[i]), int(b[i])
            diff = ai - bi - borrow
            if diff >= 0:
                res[i], borrow = str(diff), 0
            else:
                res[i], borrow = '1', 1
        return ''.join(res)

    @staticmethod
    def bit_length_from_bin(b: str) -> int:
        t = b.lstrip('0')
        return len(t) if t else 1

    @staticmethod
    def common_prefix(bit_list: list[str]) -> str:
        if not bit_list: return ""
        L = len(bit_list[0])
        for k in range(L):
            bit = bit_list[0][k]
            if any(bs[k] != bit for bs in bit_list):
                return bit_list[0][:k]
        return bit_list[0]

    @staticmethod
    def float_to_binary_unsigned(value: float, bits: int = 16) -> str:
        if value < 0: raise ValueError("value must be ≥ 0")
        i_bits = max(1, int(math.floor(math.log2(value)))+1) if value>=1 else 1
        i_bits = min(i_bits, bits)
        f_bits = bits - i_bits
        scaled = int(round(value * (2**f_bits)))
        scaled = min(scaled, (1<<bits)-1)
        return format(scaled, f'0{bits}b')

# ======================
# Noeud de trie de subsomption
# ======================
class SubsumptionTrieNode:
    def __init__(self):
        self.children: dict[str, SubsumptionTrieNode] = {}
        self.cluster_idxs: list[int] = []

# ======================
# Cluster prototype + offsets + MDL local
# ======================
class Cluster:
    def __init__(self, base: str):
        self.base = base
        self.offsets = ['0'*len(base)]
        self.size = 1

    def bits_needed(self) -> int:
        return max(BitUtils.bit_length_from_bin(off) for off in self.offsets)

    def mdl_delta_if_add(self, x: str) -> float:
        L = len(self.base)
        if x >= self.base:
            new_offs = self.offsets + [BitUtils.bin_sub(x, self.base)]
        else:
            delta = BitUtils.bin_sub(self.base, x)
            new_offs = [BitUtils.bin_sub(off, delta) for off in self.offsets] + ['0'*L]
        b_old = self.bits_needed()
        b_new = max(BitUtils.bit_length_from_bin(off) for off in new_offs)
        L_before = L + self.size * b_old
        L_after  = L + (self.size+1) * b_new
        return L_after - (L_before + L)

    def add(self, x: str):
        L = len(self.base)
        if x < self.base:
            delta = BitUtils.bin_sub(self.base, x)
            self.offsets = [BitUtils.bin_sub(off, delta) for off in self.offsets] + ['0'*L]
            self.base = x
        else:
            self.offsets.append(BitUtils.bin_sub(x, self.base))
        self.size += 1

# ======================
# Clustering incrémental + trie + grouping α
# ======================
class MDLBinaryClustering:
    def __init__(self, bits: int = 16):
        self.bits = bits
        self.clusters: list[Cluster] = []
        self.assignments: list[int] = []
        self.trie_root = SubsumptionTrieNode()

    def _trie_insert(self, base: str, idx: int):
        node = self.trie_root
        node.cluster_idxs.append(idx)
        for bit in base:
            node = node.children.setdefault(bit, SubsumptionTrieNode())
            node.cluster_idxs.append(idx)

    def _trie_remove(self, base: str, idx: int):
        node = self.trie_root
        node.cluster_idxs.remove(idx)
        for bit in base:
            node = node.children[bit]
            node.cluster_idxs.remove(idx)

    def _trie_query(self, x: str) -> list[int]:
        node = self.trie_root
        cands = node.cluster_idxs.copy()
        for bit in x:
            if bit in node.children:
                node = node.children[bit]
                cands = node.cluster_idxs.copy()
            else:
                break
        return cands

    def fit_incremental(self, data: list[float]):
        for v in data:
            x = BitUtils.float_to_binary_unsigned(v, self.bits)
            cands = self._trie_query(x)
            best_delta, best_idx = float('inf'), None
            for i in cands:
                d = self.clusters[i].mdl_delta_if_add(x)
                if d < best_delta:
                    best_delta, best_idx = d, i
            if best_idx is not None and best_delta <= 0:
                old = self.clusters[best_idx].base
                self.clusters[best_idx].add(x)
                self.assignments.append(best_idx)
                new = self.clusters[best_idx].base
                if new != old:
                    self._trie_remove(old, best_idx)
                    self._trie_insert(new, best_idx)
            else:
                idx = len(self.clusters)
                self.clusters.append(Cluster(x))
                self.assignments.append(idx)
                self._trie_insert(x, idx)

    def group_by_alpha(self, alpha: float) -> dict[str, list[int]]:
        protos = []
        for c in self.clusters:
            b_int = int(c.base,2)
            members = [format(b_int + int(off,2), f'0{self.bits}b') for off in c.offsets]
            protos.append(BitUtils.common_prefix(members))
        k = math.ceil(alpha * self.bits)
        groups = defaultdict(list)
        for idx, pat in enumerate(protos):
            groups[pat[:k]].append(idx)
        return groups

    def report_clusters(self):
        out = []
        for c in self.clusters:
            b_int = int(c.base,2)
            offs = [int(off,2) for off in c.offsets]
            members = [b_int + o for o in offs]
            out.append((b_int, members))
        return out

# ======================
# Helpers pour convertir trie → networkx
# ======================
def trie_to_nx(root: SubsumptionTrieNode) -> nx.DiGraph:
    G = nx.DiGraph()
    def dfs(node, pref):
        G.add_node(pref, count=len(node.cluster_idxs))
        for bit, child in node.children.items():
            cp = pref + bit
            G.add_edge(pref, cp)
            dfs(child, cp)
    dfs(root, "")
    return G

def hierarchy_pos(G, root='', width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5,
                  pos=None, parent=None):
    """
    Recursively positions nodes in a tree layout, top-down.
    """
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    children = list(G.successors(root))
    if len(children) != 0:
        dx = width / 2
        nextx = xcenter - width / 2
        for child in children:
            pos = hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                vert_loc=vert_loc - vert_gap, xcenter=nextx, pos=pos, parent=root)
            nextx += dx
    return pos


# ======================
# Script principal
# ======================
if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    bits  = 16
    alpha = 0.25   # <-- précision / seuil de fusion

    # Génération synthétique 3 blobs
    data = []
    for _ in range(30): data.append(min(max(random.gauss(1000,5),0),2**bits-1))
    for _ in range(30): data.append(min(max(random.gauss(5000,8),0),2**bits-1))
    for _ in range(30): data.append(min(max(random.gauss(50,3),0),2**bits-1))
    random.shuffle(data)

    # Clustering incrémental + collecte des tries
    model = MDLBinaryClustering(bits=bits)
    tries = []
    for v in data:
        model.fit_incremental([v])
        tries.append(model.trie_root)

    # Animation du trie & fusion α
    fig, ax = plt.subplots(figsize=(8,8))
    cmap = plt.get_cmap('tab10')

    def init():
        ax.clear()
        ax.set_title("Trie & clustering incrémental (0 insertions)")

    def update(frame):
        ax.clear()
        root = tries[frame]
        G    = trie_to_nx(root)

        # calcul des groupes selon alpha
        groups = model.group_by_alpha(alpha)
        # mapping cluster_idx → groupe_prefix
        idx_to_group = {}
        for pref, idxs in groups.items():
            for i in idxs:
                idx_to_group[i] = pref

        # layout
        pos = hierarchy_pos(G, root='', width=1.0, vert_gap=0.1, xcenter=0.5)


        # récupérer counts & choisir couleur
        counts = nx.get_node_attributes(G, 'count')
        node_colors = []
        for node in G.nodes():
            # chaque nœud du trie contient cluster_idxs => on prend la première
            grp = idx_to_group.get(counts[node] and counts[node] and counts[node], "")
            # simple hashing pour couleur
            node_colors.append(cmap(hash(grp) % cmap.N))

        # dessiner
        nx.draw_networkx_nodes(G, pos, ax=ax,
            node_size=[100+counts[n]*20 for n in G.nodes()],
            node_color=node_colors, edgecolors='k', alpha=0.6)
        nx.draw_networkx_edges(G, pos, ax=ax, arrows=False, alpha=0.3, width=1)
        nx.draw_networkx_labels(G, pos,
            labels={n:str(counts[n]) for n in G.nodes()},
            font_size=7)

        ax.set_axis_off()
        ax.set_title(f"Trie après {frame+1} insertions  (α={alpha})")

    anim = FuncAnimation(fig, update, frames=len(tries),
                         init_func=init, interval=700, repeat=False)
    plt.tight_layout()
    plt.show()