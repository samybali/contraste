"""
Telecom Paris : projet Contraste

Projet réalisé par : Samy Bali, Hussein El gouch, Issa Daou, Anas El Barhdadi

Encadré par Jean-Louis Dessalles
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class BitUtils:
    """Opérations bit-à-bit sur chaînes de '0'/'1'."""
    @staticmethod
    def bin_sub(a: str, b: str) -> str:
        L = len(a)
        res = ['0'] * L
        borrow = 0
        for i in range(L-1, -1, -1):
            ai, bi = int(a[i]), int(b[i])
            diff = ai - bi - borrow
            if diff >= 0:
                res[i] = str(diff)
                borrow = 0
            else:
                res[i] = '1'
                borrow = 1
        return ''.join(res)

    @staticmethod
    def bit_length_from_bin(b: str) -> int:
        trimmed = b.lstrip('0')
        return len(trimmed) if trimmed else 1

    @staticmethod
    def common_prefix(bit_list: list[str]) -> str:
        if not bit_list:
            return ""
        L = len(bit_list[0])
        for k in range(L):
            bit = bit_list[0][k]
            if any(bs[k] != bit for bs in bit_list):
                return bit_list[0][:k]
        return bit_list[0]

    @staticmethod
    def float_to_binary_unsigned(value: float, bits: int = 16) -> str:
        if value < 0:
            raise ValueError("value must be ≥ 0")
        integer_bits = max(1, int(math.floor(math.log2(value)))+1) if value >= 1 else 1
        integer_bits = min(integer_bits, bits)
        frac_bits = bits - integer_bits
        scale = 2 ** frac_bits
        scaled = int(round(value * scale))
        max_val = (1 << bits) - 1
        scaled = min(scaled, max_val)
        return format(scaled, f'0{bits}b')

class Cluster:
    """Prototype binaire avec base + offsets et calcul MDL."""
    def __init__(self, base: str):
        self.base = base
        self.offsets = ['0' * len(base)]
        self.size = 1

    def bits_needed(self) -> int:
        return max(BitUtils.bit_length_from_bin(off) for off in self.offsets)

    def mdl_delta_if_add(self, x: str) -> float:
        L = len(self.base)
        if x >= self.base:
            new_offsets = self.offsets + [BitUtils.bin_sub(x, self.base)]
        else:
            delta = BitUtils.bin_sub(self.base, x)
            new_offsets = [BitUtils.bin_sub(off, delta) for off in self.offsets] + ['0'*L]
        b_old = self.bits_needed()
        b_new = max(BitUtils.bit_length_from_bin(off) for off in new_offsets)
        L_before = L + self.size * b_old
        L_after = L + (self.size + 1) * b_new
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

class MDLBinaryClustering:
    """Clustering incrémental binaire par MDL et fusion par préfixe."""
    def __init__(self, bits: int = 16):
        self.bits = bits
        self.clusters: list[Cluster] = []
        self.assignments: list[int] = []

    def fit_incremental(self, data: list[float]):
        for value in data:
            x = BitUtils.float_to_binary_unsigned(value, self.bits)
            best_delta, best_idx = float('inf'), None
            for i, c in enumerate(self.clusters):
                d = c.mdl_delta_if_add(x)
                if d < best_delta:
                    best_delta, best_idx = d, i
            if best_idx is not None and best_delta <= 0:
                self.clusters[best_idx].add(x)
                self.assignments.append(best_idx)
            else:
                new_cluster = Cluster(x)
                self.clusters.append(new_cluster)
                self.assignments.append(len(self.clusters)-1)

    def group_by_alpha(self, alpha: float) -> dict[str, list[int]]:
        prototypes = []
        for c in self.clusters:
            base_int = int(c.base, 2)
            members = [format(base_int + int(off, 2), f'0{self.bits}b') for off in c.offsets]
            prototypes.append(BitUtils.common_prefix(members))
        k = math.ceil(alpha * self.bits)
        groups = defaultdict(list)
        for idx, pat in enumerate(prototypes):
            groups[pat[:k]].append(idx)
        return groups

    def report_clusters(self):
        results = []
        for c in self.clusters:
            base_dec = int(c.base, 2)
            offs_dec = [int(off, 2) for off in c.offsets]
            members = [base_dec + o for o in offs_dec]
            results.append((base_dec, members))
        return results

# --- Exemple d'utilisation ---
if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    bits = 16
    # Générer données synthétiques
    test2 = []
    for _ in range(30): test2.append(min(max(random.gauss(1000,5), 0), 2**bits-1))
    for _ in range(30): test2.append(min(max(random.gauss(5000,8), 0), 2**bits-1))
    for _ in range(30): test2.append(min(max(random.gauss(50,3), 0), 2**bits-1))
    random.shuffle(test2)

    # Clustering incrémental
    model = MDLBinaryClustering(bits=bits)
    model.fit_incremental(test2)

    # Fusion par alpha=0.25
    alpha = 0.25
    groups = model.group_by_alpha(alpha)

    # Tracé 1D des points et mise en évidence des clusters
    plt.figure(figsize=(10, 2))
    cmap = plt.get_cmap('tab10')
    for i, (prefix, cluster_idxs) in enumerate(groups.items()):
        # extraire valeurs associées
        xs = [v for v, c_idx in zip(test2, model.assignments) if c_idx in cluster_idxs]
        ys = [0] * len(xs)
        plt.scatter(xs, ys, s=50, alpha=0.7, label=f"{prefix}", color=cmap(i))
    plt.yticks([])
    plt.xlabel('Valeur')
    plt.title(f'Clustering 1D par préfixe (alpha={alpha})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
