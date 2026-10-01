
#Je veux générer des vecteurs de taille 2x1, tel qu'on puisse en avoir deux categories.
#Par exemple les vecteurs sont défini par le pourcentage de convexité et la taille
#On veut partitionner et faire du clustering sur ces vecteurs.
#La premiere composante de la premiere composante est peut etre très grande. La deuxieme composante du deuxièe type de vecteurs est très petits par exemple
#On veut faire du clustering sur ces vecteurs

#On va générer des vecteurs de taille
#On va générer des vecteurs de taille 2x1


import numpy as np
import matplotlib.pyplot as plt

# Définition des points dans un espace conceptuel 2D (ex: couleur, taille)
apple_red = np.array([0.9, 0.8])   # Pomme rouge (Rouge = 0.9, Taille = 0.8)
apple_green = np.array([0.2, 0.8]) # Pomme verte (Vert = 0.2, Taille = 0.8)
lemon = np.array([0.2, 0.6])       # Citron (Jaune = 0.2, Taille = 0.6)

# Fonction de contraste
def contrast(x, p):
    return x - p  # Différence vectorielle entre l'objet perçu et le prototype

# Calcul des contrastes
contrast_apple = contrast(apple_green, apple_red)  # Comparaison pomme verte vs pomme rouge
contrast_lemon = contrast(apple_green, lemon)      # Comparaison pomme verte vs citron

# Affichage des points et des vecteurs de contraste
plt.figure(figsize=(6,6))
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')

# Tracé des points
plt.scatter(*apple_red, color='red', label="Pomme Rouge")
plt.scatter(*apple_green, color='green', label="Pomme Verte")
plt.scatter(*lemon, color='yellow', label="Citron")

# Tracé des vecteurs de contraste
plt.quiver(*apple_red, *contrast_apple, angles='xy', scale_units='xy', scale=1, color='green', width=0.005, label="Contraste (Pomme verte - Rouge)")
plt.quiver(*lemon, *contrast_lemon, angles='xy', scale_units='xy', scale=1, color='blue', width=0.005, label="Contraste (Pomme verte - Citron)")

# Labels et légende
plt.xlabel("Dimension 1 (ex: Couleur)")
plt.ylabel("Dimension 2 (ex: Taille)")
plt.title("Opérateur de contraste appliqué à des fruits")
plt.legend()
plt.grid()

# Affichage du graphique
plt.show()
