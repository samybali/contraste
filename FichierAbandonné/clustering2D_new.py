import math
import random

def bin_sub(a: str, b: str) -> str:
    L = len(a)
    res = ['0'] * L
    borrow = 0
    for i in range(L - 1, -1, -1):
        ai = int(a[i])
        bi = int(b[i])
        diff = ai - bi - borrow
        if diff >= 0:
            res[i] = str(diff)
            borrow = 0
        else:
            res[i] = '1'
            borrow = 1
    return ''.join(res)

def hamming2D(a: tuple[str, str], b: tuple[str, str]) -> int:
    return sum(c1 != c2 for c1, c2 in zip(a[0], b[0])) + \
           sum(c1 != c2 for c1, c2 in zip(a[1], b[1]))

class Cluster2D:
    def __init__(self, base_x: str, base_y: str):
        self.base_x = base_x
        self.base_y = base_y
        L = len(base_x)
        self.offsets_x = ['0' * L]
        self.offsets_y = ['0' * L]
        self.size = 1

    def bits_needed(self, offsets: list[str]) -> int:
        max_off = 0
        for off in offsets:
            trimmed = off.lstrip('0')
            length = len(trimmed) if trimmed else 1
            max_off = max(max_off, length)
        return max_off

    def mdl_delta_if_add(self, x: tuple[str, str]) -> float:
        L = len(self.base_x)
        x_bin, y_bin = x

        if x_bin >= self.base_x:
            dx = bin_sub(x_bin, self.base_x)
            new_base_x = self.base_x
            new_offsets_x = self.offsets_x + [dx]
        else:
            new_base_x = x_bin
            delta_x = bin_sub(self.base_x, new_base_x)
            new_offsets_x = [bin_sub(off, delta_x) for off in self.offsets_x] + ['0' * L]

        if y_bin >= self.base_y:
            dy = bin_sub(y_bin, self.base_y)
            new_base_y = self.base_y
            new_offsets_y = self.offsets_y + [dy]
        else:
            new_base_y = y_bin
            delta_y = bin_sub(self.base_y, new_base_y)
            new_offsets_y = [bin_sub(off, delta_y) for off in self.offsets_y] + ['0' * L]

        b_old_x = self.bits_needed(self.offsets_x)
        b_old_y = self.bits_needed(self.offsets_y)
        b_new_x = self.bits_needed(new_offsets_x)
        b_new_y = self.bits_needed(new_offsets_y)

        L_before = 2 * L + self.size * (b_old_x + b_old_y)
        L_after = 2 * L + (self.size + 1) * (b_new_x + b_new_y)
        return L_after - (L_before + 2 * L)

    def add(self, x: tuple[str, str]):
        L = len(self.base_x)
        x_bin, y_bin = x

        if x_bin < self.base_x:
            new_base_x = x_bin
            delta_x = bin_sub(self.base_x, new_base_x)
            self.offsets_x = [bin_sub(off, delta_x) for off in self.offsets_x] + ['0' * L]
            self.base_x = new_base_x
        else:
            self.offsets_x.append(bin_sub(x_bin, self.base_x))

        if y_bin < self.base_y:
            new_base_y = y_bin
            delta_y = bin_sub(self.base_y, new_base_y)
            self.offsets_y = [bin_sub(off, delta_y) for off in self.offsets_y] + ['0' * L]
            self.base_y = new_base_y
        else:
            self.offsets_y.append(bin_sub(y_bin, self.base_y))

        self.size += 1

def incremental_clustering_2D(data: list[tuple[str, str]], R=5, tau=2, seuil=0.0):
    clusters: list[Cluster2D] = []
    recency: list[tuple[tuple[str, str], int]] = []
    assignments: list[int] = []

    for x in data:
        # Recherche mémoire récente par distance de Hamming
        hit = next(((idx, c_idx)
                    for idx, (x_old, c_idx) in enumerate(recency)
                    if hamming2D(x, x_old) <= tau),
                   None)

        if hit is not None:
            _, c_idx = hit
            clusters[c_idx].add(x)
            assignments.append(c_idx)
            recency.insert(0, (x, c_idx))
            recency = recency[:R]
            continue

        # Évaluation MDL
        best_delta = float('inf')
        best_idx = None
        for i, c in enumerate(clusters):
            delta = c.mdl_delta_if_add(x)
            if delta < best_delta:
                best_delta, best_idx = delta, i

        # Ajout au cluster si coût MDL raisonnable
        if best_idx is not None and best_delta <= seuil:
            c_idx = best_idx
            clusters[c_idx].add(x)
        else:
            c_idx = len(clusters)
            clusters.append(Cluster2D(*x))

        assignments.append(c_idx)
        recency.insert(0, (x, c_idx))
        recency = recency[:R]

    return clusters, assignments

# === Exemple d'utilisation ===
if __name__ == "__main__":
    L = 6  # Longueur des chaînes binaires
    N = 40  # Nombre d'éléments
    data = [
        (
            format(random.randint(0, 2 ** L - 1), f'0{L}b'),
            format(random.randint(0, 2 ** L - 1), f'0{L}b')
        )
        for _ in range(N)
    ]

    clusters, assignments = incremental_clustering_2D(data, R=5, tau=2, seuil=0.0)

    print("\n--- Instances (x,y) ---")
    for i, (x, y) in enumerate(data, 1):
        print(f"{i:2d}: ({x}, {y}) → Cluster {assignments[i - 1]}")

    print("\n--- Clusters finaux ---")
    for idx, c in enumerate(clusters):
        members = [
            (
                format(int(c.base_x, 2) + int(dx, 2), f'0{L}b'),
                format(int(c.base_y, 2) + int(dy, 2), f'0{L}b')
            )
            for dx, dy in zip(c.offsets_x, c.offsets_y)
        ]
        print(f"Cluster {idx}:")
        print(f"  Base = ({c.base_x}, {c.base_y})")
        print(f"  Membres = {members}\n")
