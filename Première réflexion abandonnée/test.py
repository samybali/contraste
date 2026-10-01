import numpy as np
import matplotlib.pyplot as plt

data= {
    "Chat": 25, "Chien": 50, "Lapin": 30, "Cheval": 160, "Éléphant": 300,
    "Girafe": 500, "Loup": 85, "Ours": 250, "Serpent": 150, "Perroquet": 45,
    "Kangourou": 140, "Lion": 120, "Tigre": 110, "Panda": 90, "Dauphin": 200,
    "Crocodile": 400, "Zèbre": 135, "Autruche": 210, "Chameau": 220, "Hippopotame": 150
}
# Fonction de clustering
def contraste_michelson(x, y):
    return 0 if x == y else abs(x - y) / (x + y)

def clustering_contraste_michelson(data, seuil_contraste):
    clusters = []  
    cluster_animaux = []  

    for animal, taille in data.items():
        meilleur_contraste = float('inf')
        meilleur_cluster = None

        for i, centre in enumerate(clusters):
            contraste = contraste_michelson(taille, centre)
            if contraste < meilleur_contraste:
                meilleur_contraste = contraste
                meilleur_cluster = i
        
        if meilleur_contraste < seuil_contraste:
            clusters[meilleur_cluster] = (clusters[meilleur_cluster] + taille) / 2
            cluster_animaux[meilleur_cluster].append((animal, taille))
        else:
            clusters.append(taille)
            cluster_animaux.append([(animal, taille)])
    
    return clusters, cluster_animaux

# Appliquer le clustering avec un seuil donné
seuil = 0.1
clusters, regroupement_animaux = clustering_contraste_michelson(data, seuil)

# 🔹 Visualisation avec Matplotlib
plt.figure(figsize=(12, 8))
colors = plt.colormaps.get_cmap("tab10")

# Espacement des clusters sur l'axe Y
y_positions = np.linspace(0, len(clusters) - 1, len(clusters))

for i, (groupe, y_pos) in enumerate(zip(regroupement_animaux, y_positions)):
    tailles = [taille for _, taille in groupe]
    noms_animaux = [animal for animal, _ in groupe]
    
    # Calcul du centre et du rayon du cluster
    centre_cluster = np.mean(tailles)
    rayon_cluster = (max(tailles) - min(tailles)) / 2 if len(tailles) > 1 else 10
    
    # Assigner une couleur unique au cluster
    color = colors(i % 10)

    # Affichage des points avec des couleurs différentes par cluster
    plt.scatter(tailles, [y_pos] * len(tailles), color=color, label=f"Cluster {i+1}")
    
    # Ajouter un cercle englobant pour chaque cluster
    circle = plt.Circle((centre_cluster, y_pos), rayon_cluster, color=color, alpha=0.2)
    plt.gca().add_patch(circle)
    
    # Ajouter des annotations (nom des animaux)
    for j, animal in enumerate(noms_animaux):
        plt.text(tailles[j], y_pos, animal, fontsize=9, verticalalignment="bottom", horizontalalignment="right")

plt.xscale("log")  # Échelle logarithmique pour mieux répartir les tailles
plt.xlabel("Taille de l'animal (cm)")  # ✅ Correction ici
plt.ylabel("Clusters")
plt.title("Clustering des animaux en fonction de leur taille (cm)")
plt.legend()
plt.grid(which="both", linestyle="--", linewidth=0.5)
plt.xlim(10, max(data.values()) + 100)  # Ajuster les limites pour une meilleure vue
plt.ylim(-1, len(clusters))  # Ajuster les limites en Y
plt.show()

""""
Explication du code 
Premièrement, on génère un jeu de données data en 1 dimension(taille des animaux)
Ensuite on definit l'opérateur de contraste, ici on a utilisé le contraste de Michelson qui est utilisé en optique pour quantifier les luminances 
maximales et minimales, Enfin, en ce qui concerne l'algorithme de Clustering, c'est un algorithme incrémental. 
Il part d'un ensemble de clusters vide, il parcours le dictionnaire du jeu de données, le premier cluster est crée et le premier élément 
du jeu de données est mis dans ce cluster, au ième élément du jeu de données, l'algorithme calcule le contraste entre cet élément et chacun des 
clusters existants et ensuite prend le minimal de ces contrastes et le compare à un contraste seuill, si ce constraste est inférieur au contraste seuil
alors l'élément est mis dans le cluster qui a le contraste minimal et le centre du cluster est mis à jour en faisant la moyenne des éléments
qui constituent le cluster, sinon il crée un nouveau cluster et met l'élément dedans. 
"""""

