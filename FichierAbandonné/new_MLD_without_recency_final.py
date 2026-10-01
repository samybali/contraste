import math
import numpy as np

# --- opérations bit-à-bit sur chaînes de '0'/'1' ---

def bin_sub(a: str, b: str) -> str:
    """Soustraction binaire a - b (a >= b), retourne bitstring."""
    L = len(a)
    res = ['0'] * L
    borrow = 0
    for i in range(L-1, -1, -1):
        ai = int(a[i])
        bi = int(b[i])
        diff = ai - bi - borrow
        if diff >= 0:
            res[i] = str(diff)
            borrow = 0
        else:
            res[i] = '1'  # diff + 2 == 1
            borrow = 1
    return ''.join(res)


def bit_length_from_bin(b: str) -> int:
    """Nombre de bits significatifs d'un bitstring."""
    trimmed = b.lstrip('0')
    return len(trimmed) if trimmed else 1

# --- définition d'un cluster binaire avec base + offsets ---

class Cluster:
    def __init__(self, base: str):
        self.base = base                  # bitstring
        self.offsets = ['0' * len(base)]  # offsets relatifs
        self.size = 1

    def bits_needed(self) -> int:
        """Nombre de bits requis pour coder le plus grand offset."""
        return max(bit_length_from_bin(off) for off in self.offsets)

    def mdl_delta_if_add(self, x: str) -> float:
        """Variation de coût MDL si on ajoute x à ce cluster."""
        L = len(self.base)

        # Cas base à conserver
        if x >= self.base:
            offset = bin_sub(x, self.base)
            new_offsets = self.offsets + [offset]
            new_base = self.base
        else:
            # Cas base à mettre à jour (x devient nouvelle base)
            new_base = x
            delta = bin_sub(self.base, new_base)
            new_offsets = [bin_sub(off, delta) for off in self.offsets] + ['0' * L]

        b_old = self.bits_needed()
        b_new = max(bit_length_from_bin(off) for off in new_offsets)

        L_before = L + self.size * b_old
        L_after = L + (self.size + 1) * b_new

        # Coût différentiel (on retire un coût de codage de base supplémentaire L)
        return L_after - (L_before + L)

    def add(self, x: str):
        """Ajoute x au cluster, avec mise à jour base et offsets si nécessaire."""
        L = len(self.base)
        if x < self.base:
            new_base = x
            delta = bin_sub(self.base, new_base)
            self.offsets = [bin_sub(off, delta) for off in self.offsets] + ['0' * L]
            self.base = new_base
        else:
            self.offsets.append(bin_sub(x, self.base))
        self.size += 1

# --- Classe de clustering incrémental pur MDL ---

class MDLClusterer:
    def __init__(self, bit_length: int):
        self.bit_length = bit_length
        self.clusters: list[Cluster] = []
        self.assignments: list[int] = []

    def add(self, x: str):
        """Ajoute un élément x au meilleur cluster ou crée un nouveau cluster."""
        best_delta = float('inf')
        best_idx = None

        for i, c in enumerate(self.clusters):
            delta = c.mdl_delta_if_add(x)
            if delta < best_delta:
                best_delta = delta
                best_idx = i

        # Si on gagne en coût MDL en l'ajoutant
        if best_idx is not None and best_delta <= 0:
            self.clusters[best_idx].add(x)
            self.assignments.append(best_idx)
        else:
            new_cluster = Cluster(x)
            self.clusters.append(new_cluster)
            self.assignments.append(len(self.clusters) - 1)

    def fit(self, data: list[str]):
        """Exécute le clustering incrémental sur une liste de bitstrings."""
        for x in data:
            self.add(x)

    def print_results(self, data: list[str]):
        """Affiche les assignments et le détail des clusters."""
        print("\n--- Instances assignées ---")
        for idx, x in enumerate(data):
            print(f"{idx+1:2d}: {x} → Cluster {self.assignments[idx]}")

        print("\n--- Clusters finaux ---")
        for i, c in enumerate(self.clusters):
            base_dec = int(c.base, 2)
            offsets_dec = [int(off, 2) for off in c.offsets]
            membres_dec = [base_dec + off for off in offsets_dec]
            print(f"Cluster {i}:")
            print(f"  Base    = {base_dec} (bin: {c.base})")
            print(f"  Offsets = {offsets_dec}")
            print(f"  Membres = {membres_dec}\n")

# --- Exemple d'utilisation ---

def float_to_int_gray(value: float, bits: int) -> str:
    """Exemple : conversion d'un float en entier puis en code Gray de longueur bits."""
    # Conversion simpliste pour l'exemple
    iv = int(round(value)) & ((1 << bits) - 1)
    gray = iv ^ (iv >> 1)
    return format(gray, f'0{bits}b')

if __name__ == "__main__":
    # Génération de données en trois clusters autour de 1000, 5000 et 50
    donnees = []
    for i in range(1000):
        if i % 3 == 0:
            donnees.append(float_to_int_gray(np.random.normal(1000, 3), 16))
        elif i % 3 == 1:
            donnees.append(float_to_int_gray(np.random.normal(5000, 4), 16))
        else:
            donnees.append(float_to_int_gray(np.random.normal(50, 5), 16))
    # Supprimer les doublons
    donnees_bin = list(set(donnees))

    # Clustering
    clusterer = MDLClusterer(bit_length=16)
    clusterer.fit(donnees_bin)
    clusterer.print_results(donnees_bin)
