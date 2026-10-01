import math

# --- opérations bit-à-bit sur chaînes de '0'/'1' ---

def bin_sub(a: str, b: str) -> str:
    """
    Soustraction binaire a - b (a >= b), bitstrings de même longueur.
    Retourne un bitstring de même longueur (sans underflow).
    """
    L = len(a)
    res = ['0'] * L
    borrow = 0
    for i in range(L-1, -1, -1):
        ai = 1 if a[i]=='1' else 0
        bi = 1 if b[i]=='1' else 0
        diff = ai - bi - borrow
        if diff >= 0:
            res[i] = '1' if diff==1 else '0'
            borrow = 0
        else:
            res[i] = '1'  # diff + 2 == 1
            borrow = 1
    return ''.join(res)

def hamming(a: str, b: str) -> int:
    """Distance de Hamming entre deux bitstrings de même longueur."""
    return sum(c1!=c2 for c1,c2 in zip(a,b))

# --- définition du cluster binaire invariant par translation ---

class Cluster:
    def __init__(self, base: str):
        self.base = base                    # bitstring de longueur L
        self.offsets = ['0'*len(base)]      # listes de bitstrings
        self.size = 1

    def bits_needed(self) -> int:
        """Nombre de bits nécessaires pour coder le plus grand offset actuel."""
        # on prend max offset en entier, mais on peut le faire sur bitstrings :
        max_off = 0
        for off in self.offsets:
            # compter bits significatifs :
            trimmed = off.lstrip('0')
            length = len(trimmed) if trimmed else 1
            max_off = max(max_off, length)
        return max_off

    def mdl_delta_if_add(self, x: str) -> float:
        """
        Calcule la variation de coût MDL si on ajoute l'instance x à ce cluster.
        Coûts :
          - Base : L bits
          - Données : size * b_old ; après ajout : (size+1) * b_new
          - Création d'un nouveau cluster coûterait L bits supplémentaires
        Renvoie ΔL = L_after - (L_before + L_creation)
        """
        L = len(self.base)
        # calculer offset hors base
        # soustraction binaire
        d = bin_sub(x, self.base) if x>=self.base else None
        new_base = self.base
        # s'il faut mettre à jour la base
        if d is None:
            new_base = x
            # recalculer tous les offsets si base change
            delta = bin_sub(self.base, new_base)
            new_offsets = [bin_sub(off, delta) for off in self.offsets] + ['0'*L]
        else:
            new_offsets = self.offsets + [d]

        # bits avant et après
        b_old = self.bits_needed()
        b_new = 0
        for off in new_offsets:
            trimmed = off.lstrip('0')
            length = len(trimmed) if trimmed else 1
            b_new = max(b_new, length)
        # coûts
        L_before = L + self.size * b_old
        L_after  = L + (self.size+1) * b_new
        # création coûte L bits
        return (L_after) - (L_before + L)

    def add(self, x: str):
        """Ajoute définitivement x au cluster (met à jour base et offsets)."""
        # même logique que mdl_delta_if_add pour mettre à jour
        L = len(self.base)
        if x < self.base:
            new_base = x
            delta = bin_sub(self.base, new_base)
            self.offsets = [bin_sub(off, delta) for off in self.offsets] + ['0'*L]
            self.base = new_base
        else:
            d = bin_sub(x, self.base)
            self.offsets.append(d)
        self.size += 1

# --- clustering incrémental avec FIFO de récence ---

def incremental_clustering(data: list[str],
                           R: int=5,
                           tau: int=0) -> tuple[list[Cluster], list[int]]:
    """
    data : liste de bitstrings
    R   : taille du buffer FIFO de récence
    tau : seuil de Hamming pour rattachement instantané
    """
    clusters: list[Cluster] = []
    recency: list[tuple[str,int]] = []  # buffer FIFO de (instance, cluster_idx)
    assignments: list[int] = []

    for x in data:
        # 1) rattachement direct si instance récente similaire
        hit = next(((idx,c_idx) 
                    for idx,(x_old,c_idx) in enumerate(recency)
                    if hamming(x, x_old) <= tau),
                   None)
        if hit is not None:
            _, c_idx = hit
            clusters[c_idx].add(x)
            assignments.append(c_idx)
            # mettre à jour FIFO
            recency.insert(0,(x,c_idx))
            recency = recency[:R]
            continue

        # 2) sinon, évaluer MDL pour chaque cluster
        best_delta = float('inf')
        best_idx   = None
        for i,c in enumerate(clusters):
            delta = c.mdl_delta_if_add(x)
            if delta < best_delta:
                best_delta, best_idx = delta, i

        # 3) décision
        L = len(x)
        if best_idx is not None and best_delta <= 0:
            c_idx = best_idx
            clusters[c_idx].add(x)
        else:
            # nouveau cluster
            c_idx = len(clusters)
            clusters.append(Cluster(x))

        assignments.append(c_idx)
        # 4) mise à jour FIFO
        recency.insert(0,(x,c_idx))
        recency = recency[:R]

    return clusters, assignments

# --- Exemple d’utilisation modifié ---

if __name__ == "__main__":
    import random

    # Paramètres
    L = 8           # longueur des bit-strings
    N = 50           # nombre total d’instances à générer
    R = 5            # taille du buffer de récence
    tau = 1          # seuil de Hamming pour rattachement direct

    # Générer N instances aléatoires en 8 bits
    data = [format(random.randint(0, 2**L-1), f'0{L}b') for _ in range(N)]

    # Clustering incrémental
    clusters, assignments = incremental_clustering(data, R=R, tau=tau)

    # Affichage des résultats
    print("\n--- Instances générées ---")
    for i, x in enumerate(data, 1):
        print(f"{i:2d}: {x} → Cluster {assignments[i-1]}")

    print("\n--- Clusters finaux et prototypes associés ---")
    for idx, c in enumerate(clusters):
        # reconstruire les membres à partir de base + offsets
        membres = [
            format(int(c.base,2) + int(off,2), f'0{L}b')
            for off in c.offsets
        ]
        print(f"Cluster {idx}:")
        print(f"  Base = {c.base}")
        print(f"  Offsets = {c.offsets}")
        print(f"  Membres = {membres}\n")

