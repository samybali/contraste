import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Définition des points dans un espace conceptuel 3D (couleur, taille, acidité)
apple_red = np.array([0.9, 0.8, 0.3])   # Pomme rouge
apple_green = np.array([0.2, 0.8, 0.4]) # Pomme verte
lemon = np.array([0.2, 0.6, 0.9])       # Citron

# Fonction de contraste
def contrast(x, p):
    return x - p  # Différence vectorielle entre l'objet perçu et le prototype

# Calcul des vecteurs de contraste
contrast_apple = contrast(apple_green, apple_red)  # Pomme verte - Pomme rouge
contrast_lemon = contrast(apple_green, lemon)      # Pomme verte - Citron

# Points de départ et d'arrivée pour chaque vecteur
start_apple = apple_red
end_apple = apple_red + contrast_apple

start_lemon = lemon
end_lemon = lemon + contrast_lemon

# Création d'une figure 3D
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Tracé des points
ax.scatter(*apple_red, color='red', s=100, label="Pomme Rouge")
ax.scatter(*apple_green, color='green', s=100, label="Pomme Verte")
ax.scatter(*lemon, color='yellow', s=100, label="Citron")

# Tracé des lignes de contraste
# Contraste Pomme verte - Pomme rouge (ligne verte)
ax.plot([start_apple[0], end_apple[0]],
        [start_apple[1], end_apple[1]],
        [start_apple[2], end_apple[2]],
        color='green', label="Contraste (Pomme verte - Rouge)")

# Contraste Pomme verte - Citron (ligne bleue)
ax.plot([start_lemon[0], end_lemon[0]],
        [start_lemon[1], end_lemon[1]],
        [start_lemon[2], end_lemon[2]],
        color='blue', label="Contraste (Pomme verte - Citron)")

# (Optionnel) Marqueurs pour la fin de la ligne
ax.scatter(end_apple[0], end_apple[1], end_apple[2], color='green', marker='^', s=70)
ax.scatter(end_lemon[0], end_lemon[1], end_lemon[2], color='blue', marker='^', s=70)

# Configuration des axes
ax.set_xlabel("Dimension 1 (ex: Couleur)")
ax.set_ylabel("Dimension 2 (ex: Taille)")
ax.set_zlabel("Dimension 3 (Acidité)")
ax.set_title("Opérateur de contraste en 3D (sans quiver)")
ax.legend()
plt.show()
