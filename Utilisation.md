# Guide d'utilisation - Algorithmes de Clustering pour Artishow Contraste

## Vue d'ensemble du projet

Le projet **Artishow Contraste** vise à développer des algorithmes de clustering basés sur le concept de **contraste** (C = O - P, où O est l'observation et P est le prototype/attente). L'objectif est de détecter des anomalies et de catégoriser des données en identifiant ce qui "sort de l'ordinaire".

## À quoi servent ces codes ?

Ces deux implémentations proposent des approches différentes pour le clustering de données :

1. **`clustering_choix_specifique.py`** : Clustering 1D optimisé avec compression MDL
2. **`tree.py`** : Clustering 2D hiérarchique avec visualisation arborescente

Les deux approches permettent de :
- Identifier automatiquement des groupes dans vos données
- Détecter les points isolés (anomalies potentielles)
- Visualiser les résultats de manière intuitive
- Comprimer l'information tout en préservant les patterns importants

---

## 1. clustering_choix_specifique.py - Clustering 1D avec MDL

### Principe de fonctionnement

Cet algorithme utilise une approche **incrémentale** basée sur le principe MDL (Minimum Description Length) :

1. **Représentation binaire** : Chaque valeur est convertie en binaire
2. **Clustering incrémental** : Les données sont ajoutées une par une
3. **Optimisation MDL** : Minimise le coût de description des clusters
4. **Fusion par préfixe** : Regroupe les clusters similaires

### Comment l'utiliser

```python
import numpy as np
from clustering_choix_specifique import MDLBinaryClustering

# 1. Créer une instance du modèle
bits = 16  # Précision de la représentation binaire
model = MDLBinaryClustering(bits=bits)

# 2. Préparer vos données (valeurs numériques 1D)
data = [100.5, 102.3, 98.7,  # Groupe 1
        500.2, 498.9, 501.1,  # Groupe 2
        50.1, 48.5, 51.3]     # Groupe 3

# 3. Effectuer le clustering
model.fit_incremental(data)

# 4. Obtenir les résultats
clusters = model.report_clusters()
for base, members in clusters:
    print(f"Cluster avec base {base}: {members}")

# 5. Fusionner les clusters similaires (optionnel)
alpha = 0.25  # Paramètre de fusion (0 à 1)
groups = model.group_by_alpha(alpha)
```

### Paramètres importants

- **`bits`** : Nombre de bits pour la représentation (8-32)
  - Plus de bits = plus de précision
  - Valeur recommandée : 16
  
- **`alpha`** : Contrôle la fusion des clusters (0-1)
  - 0 = pas de fusion
  - 1 = fusion maximale
  - Valeur recommandée : 0.25

### Exemple de visualisation

Le code génère automatiquement un graphique 1D montrant les clusters identifiés avec des couleurs différentes.

---

## 2. tree.py - Clustering 2D Hiérarchique

### Principe de fonctionnement

Cet algorithme construit un **arbre hiérarchique** pour des données 2D :

1. **Construction incrémentale** : L'arbre grandit à mesure que les données arrivent
2. **Patterns avec wildcards** : Utilise des étoiles (*) pour généraliser
3. **Élagage MDL** : Supprime les nœuds inutiles
4. **Visualisation riche** : Affiche l'arbre et les clusters

### Comment l'utiliser

```python
from tree import BinaryHierarchicalTree

# 1. Créer un arbre
tree = BinaryHierarchicalTree(n_bits=4)  # 4 bits = valeurs 0-15

# 2. Insérer des données 2D
data_points = [
    (10, 12),  # Point 1
    (11, 13),  # Point 2
    (2, 3),    # Point 3
    (14, 14)   # Point 4
]

for point in data_points:
    tree.insert(point)

# 3. Visualiser l'arbre complet
tree.visualize()

# 4. Appliquer l'élagage MDL
tree.apply_mdl_pruning(verbose=True)

# 5. Visualiser les clusters finaux
tree.visualize_clusters_2d(show_centers=True)

# 6. Obtenir les assignments
clusters = tree.get_cluster_assignments()
for cluster_id, (node, points) in clusters.items():
    if node:  # Cluster normal
        print(f"Cluster {node.pattern}: {points}")
    else:  # Point isolé
        print(f"Point isolé: {points[0]}")
```

### Paramètres importants

- **`n_bits`** : Nombre de bits par dimension
  - 4 bits = valeurs 0-15
  - 8 bits = valeurs 0-255
  - Plus de bits = plus grande plage de valeurs

- **`verbose`** : Active les messages détaillés lors de l'élagage

### Visualisations disponibles

1. **`visualize()`** : Affiche l'arbre hiérarchique complet
2. **`visualize_with_mdl_scores()`** : Affiche l'arbre avec les scores MDL
3. **`visualize_clusters_2d()`** : Affiche les clusters en 2D avec couleurs
4. **`print_tree()`** : Affiche la structure textuelle

---

## Cas d'usage pratiques

### 1. Détection d'anomalies dans des mesures

```python
# Températures relevées
temperatures = [20.1, 20.5, 19.8, 20.2,  # Normal
                35.2,  # Anomalie !
                20.0, 19.9, 20.3]

model = MDLBinaryClustering(bits=16)
model.fit_incremental(temperatures)
clusters = model.report_clusters()

# Les clusters avec un seul membre sont potentiellement des anomalies
for base, members in clusters:
    if len(members) == 1:
        print(f"Anomalie détectée : {members[0]}")
```

### 2. Groupement de coordonnées GPS

```python
# Coordonnées simplifiées (x, y)
tree = BinaryHierarchicalTree(n_bits=8)

locations = [
    (100, 120), (102, 118), (99, 121),  # Zone A
    (200, 50), (198, 52), (201, 49),    # Zone B
    (150, 150)  # Point isolé
]

for loc in locations:
    tree.insert(loc)

tree.apply_mdl_pruning()
tree.visualize_clusters_2d()
```

### 3. Analyse de patterns temporels

```python
# Heures d'arrivée (converties en minutes depuis minuit)
arrivals = [480, 485, 490,  # ~8h00
            540, 545,        # ~9h00
            720, 725, 730]   # ~12h00

model = MDLBinaryClustering(bits=12)
model.fit_incremental(arrivals)
groups = model.group_by_alpha(0.3)
```

---

## Conseils d'utilisation

### Pour clustering_choix_specifique.py :
- Idéal pour des données 1D avec des groupes bien séparés
- Très rapide pour traiter de grandes quantités de données
- Ajustez `alpha` pour contrôler la granularité des clusters
- Attention : Les données doivent être positives

### Pour tree.py :
- Parfait pour visualiser la hiérarchie des données
- Gère bien les données 2D avec relations complexes
- L'élagage MDL simplifie automatiquement la structure
- Attention : Plus lent sur de très grandes quantités de données

---

## Alternatives et extensions

### 1. Adapter à d'autres dimensions

Pour tree.py, vous pouvez facilement l'étendre à 3D ou plus :

```python
# Modifier la classe Node pour accepter des tuples de n'importe quelle taille
pattern: Tuple[str, ...]  # Au lieu de Tuple[str, str]
```

### 2. Clustering en temps réel

Les deux algorithmes supportent l'ajout incrémental, parfait pour :
- Flux de données en temps réel
- Apprentissage en ligne
- Systèmes adaptatifs

---

## Pour aller plus loin

1. **Comprendre le contraste** : C = O - P
   - O = Observation (ce qu'on voit)
   - P = Prototype (ce qu'on attend)
   - C = Contraste (la surprise)

2. **MDL (Minimum Description Length)**
   - Principe : la meilleure explication est la plus courte
   - Application : garder les clusters qui compriment le mieux les données

3. **Applications du projet**
   - Détection d'anomalies
   - Catégorisation pertinente
   - Négation (identifier ce qui n'est PAS dans un groupe)

---

## Résolution de problèmes

**Erreur "value must be ≥ 0"** : Normalisez vos données pour qu'elles soient positives

**Trop de clusters** : Augmentez le paramètre `alpha` ou réduisez `n_bits`

**Visualisation illisible** : Utilisez `figsize` plus grand ou filtrez les données

**Mémoire insuffisante** : Traitez les données par batch ou réduisez `n_bits`