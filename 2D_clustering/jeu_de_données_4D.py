"""
Telecom Paris : projet Contraste

Projet réalisé par : Samy Bali, Hussein El gouch, Issa Daou, Anas El Barhdadi

Encadré par Jean-Louis Dessalles
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Créer un jeu de données d'animaux avec 4 dimensions
# Dimensions: 
# 1. Taille (hauteur/longueur en mètres)
# 2. Poids (en kg)
# 3. Vitesse maximale (en km/h)
# 4. Espérance de vie (en années)

# Créer le jeu de données
animals_data = {
    'nom': [
        'Éléphant d\'Afrique', 'Lion', 'Girafe', 'Guépard', 'Gorille', 
        'Aigle royal', 'Dauphin commun', 'Baleine bleue', 'Tortue géante', 'Colibri',
        'Requin blanc', 'Crocodile du Nil', 'Autruche', 'Chimpanzé', 'Koala',
        'Kangourou', 'Paresseux', 'Perroquet', 'Tigre', 'Loup',
        'Ours polaire', 'Léopard', 'Rhinocéros', 'Hippopotame', 'Chauve-souris',
        'Panda géant', 'Caméléon', 'Otarie', 'Manchot empereur', 'Serpent anaconda'
    ],
    'taille': [
        3.3, 1.2, 5.5, 0.9, 1.7,
        0.9, 2.5, 25.0, 1.3, 0.1,
        6.0, 5.0, 2.5, 1.2, 0.8,
        2.0, 0.6, 0.3, 3.0, 1.5,
        2.5, 1.9, 3.8, 4.5, 0.15,
        1.8, 0.4, 2.5, 1.2, 8.0
    ],
    'poids': [
        6000, 190, 1600, 65, 200,
        6.5, 80, 150000, 300, 0.02,
        2000, 700, 150, 60, 13,
        85, 8, 1, 300, 80,
        450, 90, 2300, 3000, 0.03,
        150, 0.2, 300, 40, 250
    ],
    'vitesse': [
        40, 80, 60, 120, 40,
        240, 60, 50, 10, 80,
        56, 30, 70, 45, 10,
        70, 0.3, 35, 65, 75,
        40, 58, 45, 30, 100,
        32, 0.5, 40, 9, 8
    ],
    'esperance_vie': [
        70, 15, 25, 12, 40,
        30, 35, 90, 150, 3,
        70, 60, 40, 50, 15,
        12, 20, 50, 15, 14,
        25, 17, 40, 45, 20,
        20, 8, 23, 20, 30
    ],
    'categorie': [
        'Mammifère terrestre', 'Mammifère terrestre', 'Mammifère terrestre', 'Mammifère terrestre', 'Mammifère terrestre',
        'Oiseau', 'Mammifère marin', 'Mammifère marin', 'Reptile', 'Oiseau',
        'Poisson', 'Reptile', 'Oiseau', 'Mammifère terrestre', 'Mammifère terrestre',
        'Mammifère terrestre', 'Mammifère terrestre', 'Oiseau', 'Mammifère terrestre', 'Mammifère terrestre',
        'Mammifère terrestre', 'Mammifère terrestre', 'Mammifère terrestre', 'Mammifère terrestre', 'Mammifère terrestre',
        'Mammifère terrestre', 'Reptile', 'Mammifère marin', 'Oiseau', 'Reptile'
    ]
}

# Créer un DataFrame Pandas
df = pd.DataFrame(animals_data)

# Afficher les 5 premières lignes du jeu de données
print("Jeu de données d'animaux à 4 dimensions:")
print(df.head())

# Sauvegarder le jeu de données dans un fichier CSV
df.to_csv('animal_dataset_4d.csv', index=False)
print("Jeu de données sauvegardé dans 'animal_dataset_4d.csv'")

# Visualisation 3D des 3 premières dimensions avec la 4ème en couleur
# Utiliser une échelle logarithmique pour le poids qui a une grande variation
df['log_poids'] = np.log10(df['poids'])

# Créer le graphique 3D
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Tracer le nuage de points 3D
scatter = ax.scatter(
    df['taille'], 
    df['log_poids'],
    df['vitesse'],
    c=df['esperance_vie'],
    cmap='viridis',
    s=100,
    alpha=0.7
)

# Ajouter les étiquettes pour certains animaux
for i, txt in enumerate(df['nom']):
    if i % 3 == 0:  # N'afficher qu'une partie des étiquettes pour plus de lisibilité
        ax.text(df['taille'][i], df['log_poids'][i], df['vitesse'][i], txt, size=8)

# Ajouter les légendes et les titres
ax.set_xlabel('Taille (m)')
ax.set_ylabel('Log Poids (kg)')
ax.set_zlabel('Vitesse (km/h)')
ax.set_title('Visualisation 3D des animaux')

# Ajouter une barre de couleur pour l'espérance de vie (4ème dimension)
cbar = plt.colorbar(scatter)
cbar.set_label('Espérance de vie (années)')

plt.tight_layout()
plt.show()

# Créer aussi une matrice de nuages de points pour visualiser toutes les dimensions
plt.figure(figsize=(12, 10))
sns.pairplot(df[['taille', 'poids', 'vitesse', 'esperance_vie', 'categorie']], 
             hue='categorie', 
             diag_kind='kde',
             plot_kws={'alpha': 0.6, 's': 80})
plt.suptitle('Matrice de nuages de points pour les 4 dimensions', y=1.02, fontsize=16)
plt.tight_layout()
plt.show()