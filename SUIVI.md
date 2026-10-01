# Suivi de projet Artishow Contraste

## 14/02/2025

L'encadrant a présenté le projet dans son **ensemble**, détaillant les objectifs, les attentes et les défis potentiels. Les membres de l'équipe ont étudié plusieurs documents de référence, notamment :

- Le paradoxe de Kleinberg
- Des articles scientifiques sur la détection d'anomalies
- Différentes approches de clustering

## 17/02/2025
**Hussein et Samy** ont mené une discussion approfondie sur :

- La stratégie de mise en œuvre du projet
- Les aspects techniques de l'implémentation
- La structure du déroulement du projet
- Les objectifs à atteindre

## 18/02/2025
**L'équipe complète** s'est réunie pour :

- Établir le planning du projet
- Échanger sur les documents bibliographiques étudiés
- Discuter de l'organisation générale du projet et de son déroulement prévu

## 20/02/2025 et 21/02/2025
**Hussein** :

- Travail de bibliographie approfondie sur les articles : https://perso.telecom-paristech.fr/jld/papers/Dessalles_14050201.pdf et https://gsileno.net/articles/AIC2018.pdf
- Les enjeux et direction à prendre pour le projet se sont décantés. Nous comprenons que l'opérateur de contratse sera central ici : C = O - P;
exemple "This girl has a red face" O = gril's face , P= average girl face , C likely to fall into the red region if conceptual spaces include colour
- mini implémentation en 2D puis 3D avec un citron, une pomme verte et une pomme rouge. 2D = taille x coleur; 3D = taille x couleur x acidité. Cf le code python

**Issa** : 
- Implémentation du contraste de Michelson en 1D avec un set d'animeux cf le code python

Planning Finalisé

## 22/02/2025
**Samy** : Conception d'un algorithme de clustering en 1D en utilisant la distance de hamming et en génerant des nombres aléatoires avec des lois gaussiennes. Les distances sont représentées dans une matrice. On souhaite se débarasser de la distance, il faut donc changer l'approche.

## 10/03/2025
**Hussein**: J'ai exploré la distance pondéré qui est bien accomodante pour des valeurs comme 32 (0100000) et 33 (0100001) mais ne l'est pas du tout pour
64 et 63. Je reviens donc sur la distance de Hamming classique couplé au code de Gray. Je ferai le test demain

**Anas** : J'ai commencé à travailler sur la représentation des données en utilisant un arbre binaire de recherche (BST) pour organiser les points en fonction de leur distance binaire. Cela devrait faciliter la recherche des points les plus proches et améliorer l'efficacité du clustering.


**Samy** : Deuxième tentative de créer un sytème de clustering. Cette fois ci je crée des protoypes en remplacer les bits différents pour des nobmres proches par des étoiles. Ensuite je ne garde que les prototypes qui couvrents beacoup d'instances.

## 11/03/2025
**Hussein**: j'ai essayé plusieurs méthodes et explorer des techniques pour représenter des nombres tout en faisant en sorte qu'il n'y ait pas de problème tel que la représentation
de 63 et 64 changent du tout au tout, donc la distance de Hamming et de 7. Pourtant on a envie de dire qu'ils sont proches

Ce que j'ai fait : j'ai utilisé le code de gray pour bien que la distance de hamming entre deux entiers consécutifs soit de 1.
Je teste donc un clustering incrémental qui utilise un seuil (voir le 1D_clust_gaussienne.py)

Je trouve ce test assez concluant puisque pour 3 tirages gaussiens assez espacés, avec un seuil égal à 2 on trouve 4 clusters. Voici le résultat du code : 
Nombre de clusters trouvés : 4

Cluster 0: size=51
Prototype (Gray) = 00001101
Quelques valeurs = [6.473919689274532, 21.625051644788396, 13.704556369017876, 7.974337759331153, 10.417727190009511, 9.073164614375075, 21.95109025424472, 12.931297537843108, 21.643805008750448, 14.76050979182051] ...

Cluster 1: size=28
Prototype (Gray) = 00110000
Quelques valeurs = [32.27183763561578, 30.3480097068016, 33.73806403170392, 32.24172729729859, 31.03863016074717, 34.516142865001974, 31.813088858821686, 33.489573037651404, 31.493965887141403, 35.29895409526292] ...

Cluster 2: size=49
Prototype (Gray) = 00011110
Quelques valeurs = [18.70976049078975, 19.560656224324976, 22.955788089483033, 10.628494665190548, 20.64816793878959, 20.17409413647634, 20.723272050095268, 20.722791211016826, 20.65750221931937, 19.855979756839332] ...

Cluster 3: size=22
Prototype (Gray) = 00110111
Quelques valeurs = [36.760571281400814, 37.34546861533193, 35.52373343849552, 37.25579909806032, 37.44055165210901, 35.90464202700084, 37.37309584112914, 42.389726337455855, 35.68237980381239, 38.428468443545064] ...

## 17/03/2025
**Samy** : Tavail sur le clustering de données en 1D :

- Travail sur l'importance à donner aux bits
- Création de prototypes étapes par étapes
- Étude du choix à effectuer sur les prototypes pour ne garder que ceux qui semblent pertinents

**Issa** : Travail sur le clustering en 2D

- Génération de vecteurs aléatoires en dimension 2
- Visualisation des données via matplotlib. 
- Implémentation d'un code basé sur clustering incrémental

## 18/03/2025

**Samy**
-  Début de l'organisation du code. Ajout de classes pour structurer le code et clarifier l'utilisation des données :
    - Création d'une classe Data pour représenter les données à traiter
    - Création d'une classe Prototype pour représenter les prototypes
    - Création de la classe Tree pour représenter les liens qui existents entre les différents prototypes
    - Définition de deux variables globales NB_BITS et BITS_IGNORED qui permettent de définir le nobmre de bits sur lesquelles le travail sera effectué

- Avancée sur le clustering de données à une dimension :
    - Travail sur le choix des prototypes intéressant ou non : On classe les prototypes dans une liste en fonction du nombre d'instances couvertes par le prototype. De plus il peut exister un lien entre deux éléments "lointains" dans la liste grâce à la structure d'arbre. Si deux prototypes sont éloignés dans la liste mais qu'ils sont liés par l'arbre alors on a une donnée intéressante garder. Ce qui semble encore assez compliqué c'est l'arbitrage de ce qui est loin ou pas. Il me semble qu'il est nécessaire de faire une comparaison de ces informations entre elles mais je n'arrive pas encore à obtenir le bon paramétrage.

**Hussein**
- Recherche de pleins de façons différentes de faire du clustering en 1D (silhouette, utilisation du 1er et 3eme quartile, variance) tout en ayant un seuil adaptatif -> mauvaise piste
**Issa**
-Continuer avec le clustering 2D en intégrant une méthode basé sur la concatenation des codes de gray de la coordonnées en x et en y. 

## 23/03/2025

**Anas**
- Implémentation d'un arbre binaire de recherche (BST) pour accélérer la recherche des points les plus proches dans l'espace binaire :
  - Création de la classe BinarySearchTree qui organise les données selon leur représentation binaire
  - Implémentation d'une méthode d'insertion efficace basée sur les bits de la représentation binaire
  - Optimisation de la recherche du point le plus proche en utilisant la distance de Hamming
  - L'algorithme explore d'abord la branche la plus prometteuse puis vérifie l'autre branche seulement si nécessaire
  - Cette structure accélère considérablement la recherche des voisins proches par rapport à une recherche linéaire

- Réflexion sur l'intégration de cette structure dans l'algorithme de clustering :
  - Le BST permet de trouver rapidement les points similaires dans l'espace des caractéristiques
  - Cette approche devrait améliorer la scalabilité de l'algorithme sur des grands jeux de données
  - La prochaine étape sera de tester cette structure avec différents paramètres de seuil pour la formation des clusters

## 24/03/2025
**Hussein**
- Exploration de méthode de clsutering via l'utilisation de silhouette __> mauvaise piste 

## 25/03/2025

**Samy**

- Discussion avec l'encadrant :
  - Définiton de la manière d'utiliser les arbres pour représenter les adresses que l'on souhaite associer aux instances
  - Travail sur une manière d'associer une instance au prototype le plus proche : On choisit l'adresse la plus petite (en nombre de bits) parmi celles données parmi les trois manières : pile, arbre et liste par score décroissant
  - Discussion sur le choix d'un seuil sur la taille des adresses pour ne travailler que sur des prototypes qui semblent intéressants

**Hussein**
- Discussion avec l'encadrant :
  - Basculement d'une méthode de clustering à plutôt à un moyen de représentation via SGG les nombres ex: 33 et 37 en binaire sont subsumés par 100***

## 31/03/2025

**Hussein**
- Réalisation d'un code réalisant la "subsumation" des représentation bnaire au fur et à mesure sans prendre en compte des 
possible problèmes ex : 7 et 8
- Créer une hiérarchie de schémas de généralisation (SGG) à partir de représentations binaires d'une liste de nombres, 
en appliquant progressivement la subsumation (i.e. regrouper des binaires similaires sous un schéma plus général avec des *).

**Samy**
- Travail de recherche sur les arbres radix:
  - Un arbre radix (PATRICIA tree) est une arbre permettant de représenter un ensemble de mots(donc dans notre cas les prototypes) dans le but but de faciliter la recherche.
  - Dans notre cas, l'utilisation d'un arbre radix peut permettre d'effectuer la recherche du prototype le plus proche. Pour trouver un mot on regarde à chaque noeud de l'arbre (en partant de la racine) si c'est un préfixe du mot recherché ou non jusqu'à arriver à une feuille et on regarde si la feuille donnée correspond au mot recherché. Cependant, cette méthode nécessite d'aller jusqu'au niveau des feuilles. Or, nous recherchons des prototypes qui peuvent être dans des noeuds.
- Travail sur la recherche du prototype le plus proche:
  - Mise en place de la première methode : recherche du prototype le plus proche grâce à un parcours d'arbre
  - Réflexion sur la manière de représenter le prototype qu'on souhaite associer à l'instance : un dictionnaire semble être adapté pour une manipulation facilitée par les autres membres de l'équipe
**Issa** 
- Réflexion sur le problème de la diagonale lors du clustering en 2D avec 2 clusters.
- Exploration des arbres radix.  
## 01/04/2025
**Hussein** 

Présentation du graphe finalisé ssg.py à l'encadrant et validation.
Remarque de sa part :
Essayer de le recentrer, les problèmes de saut en bits (ex de 7 à8 ou 63 à 64) seront à régler après ("avec un patch"et ne mettent pas en danger le principe général.
Relancer Issa sur le problème de la diagonale en  2D 
## 08/04/2025

**Issa et Anas**

Réflexion sur la manière d'appréhender le problème de clustering en 2D : 

- Nous abordons le clustering 2D comme une extension du clustering 1D, en traitant chaque composante (x et y) comme un problème indépendant. Cette approche permet d'appliquer nos méthodes de clustering 1D sur chaque dimension, mais présente une difficulté : elle ne capture pas les relations entre les dimensions dans l'espace 2D. 

- Lors du traitement incrémental, le premier vecteur ajouté définit un cluster avec un prototype idéal pour x et y. La complexité survient quand nous avons plusieurs clusters, car un vecteur peut être proche d'un cluster selon sa composante x, mais plus proche d'un autre selon sa composante y. 

- Pour résoudre ce dilemme, nous proposons une méthode de pondération des distances : 
  1. Pour le cluster optimal en x : nous fixons sa valeur x et calculons l'écart avec sa valeur y, obtenant ainsi un premier poids
  2. Pour le cluster optimal en y : nous fixons sa valeur y et calculons l'écart avec sa valeur x, obtenant un second poids
  3. Nous sélectionnons ensuite le cluster ayant le poids minimal, représentant la meilleure correspondance globale

  
**Hussein**

Réalisation d'un jeu de données en 4D,que je vais enrichir 

## 10/04/2025

**Samy, Issa et Hussein**

Visio avec l'encadrant pour préciser quelque points:
en 2D : Issa a compris le problème de la diagonale
en 1D : Notre système doit contenir trois mémoires : - une épisodique (file qui voit les objets arrivé)
                                                     - graphe
                                                     - mémoire de score (protoype les plus généraux ont les plus grands scores

                                                     

## 11/04/2025

**Samy**

- Clarification des différents algorithmes discutés avec l'encadrant: il ya avait encore certaines zones de flous notamment sur ce qui est défini par l'adresse depuis le prototype pour le calcul du prototype le plus proche.

## 12/04/2025

**Anas**

- Implémentation complète de trois approches de clustering en 2D dans le fichier Clustering_2D_V22.py :

  1. **Clustering avec Distance Euclidienne** :
     - Utilisation de la distance euclidienne classique entre les points et les centres des clusters
     - Approche intuitive visuellement, car elle correspond à la perception géométrique de l'espace
     - Avantage : simplicité et interprétabilité des résultats
  
  2. **Clustering avec Distance de Hamming Pondérée** :
     - Implémentation d'une distance de Hamming où chaque bit a un poids de 2^position
     - Les bits de poids fort ont une influence beaucoup plus importante que les bits de poids faible
     - Permet de mieux capturer la similitude sémantique entre les représentations binaires
  
  3. **Clustering par Composantes** :
     - Approche avancée traitant séparément les composantes x et y avec leurs prototypes respectifs
     - Processus décisionnel sophistiqué :
       - Recherche de correspondances de prototypes pour x et y parmi les clusters existants
       - Gestion de cinq cas différents (aucune correspondance, correspondance pour x seulement, pour y seulement, correspondance commune, ou conflit)
       - Résolution des conflits (x→cluster A, y→cluster B) via la méthode de pondération des distances
     - Cette approche résout précisément le problème identifié dans notre dernière réflexion du 08/04

- **Visualisation comparative** :
  - Implémentation d'une visualisation côte à côte des trois méthodes pour faciliter la comparaison
  - Utilisation de couleurs aléatoires distinctes pour chaque cluster
  - Annotation des clusters avec leurs prototypes binaires (montrant les bits significatifs et les '*')

- **Analyse et évaluation** :
  - La méthode par composantes donne des résultats particulièrement intéressants pour nos données
  - Elle permet de capturer à la fois les similarités binaires et les relations spatiales entre les points
  - Les tests montrent que l'approche par composantes résout efficacement les cas ambigus où x et y suggèrent des clusters différents

## 13/04/2025

**Anas**

- Réalisation de tests approfondis sur les trois méthodes de clustering implémentées :

  1. **Tests avec un nombre accru de points** :
     - Augmentation du nombre de points de test de 30 à 100, puis à 500
     - Observation de l'impact sur les performances et la stabilité des algorithmes
  
  2. **Expérimentation avec plus de groupes** :
     - Extension des tests à 4 groupes (au lieu de 3) positionnés stratégiquement
     - Configuration des points pour créer des cas de diagonale complexes
  
  3. **Tests de configuration des paramètres** :
     - Variation du nombre de bits significatifs (fixé à 4)
     - Modification des seuils de distance pour la formation des clusters
     - Analyse de la sensibilité de chaque méthode aux changements de paramètres

**Hussein**

Travail de bibliographie sur MDL ideal comme l'encadrant me l'avait conseillé

**Samy**

- Avancement sur le code qui choisit le prototype le plus proche:
  - Génération de tous les prototypes jusqu'à l'instance la plus général
  - Écriture de l'algorithme  pour l'association par score 

## 14/04/2025

**Anas**

- Amélioration de la représentation binaire des nombres :
  - Implémentation d'un mécanisme adaptatif pour maximiser l'utilisation des 20 bits disponibles
  - Prise en compte des valeurs décimales et négatives dans la conversion vers le code Gray

- Extension des tests à des distributions de données plus complexes :
  - Expérimentation avec divers positionnements des centres de clusters
  - Tests avec différentes variances pour explorer la robustesse du clustering
  - Analyse de l'impact des valeurs extrêmes sur la formation des prototypes

**Hussein**

**Principe MDL** : choisir le modèle (ensemble de prototypes) qui minimise la somme du coût de sa description et celle des données qu’il encode.
**Affectation des instances** : chaque nouvelle instance est rattachée au prototype qui entraîne le plus faible surcoût de description (coût modèle + données).
** Prototypes binaires généralisés** : représentés avec des jokers (*), leur complexité dépend du nombre de bits fixés vs généralisés.
**Évolution hiérarchique** : la structure en graphe permet une compression efficace en regroupant les régularités à différents niveaux.
**Compromis optimal** : MDL guide la spécialisation ou la généralisation des prototypes en fonction du gain global de compression.

## 15/04/2025


**Anas**

- Analyse approfondie du défi du clustering 2D : identification d'une problématique cruciale concernant les vecteurs qui présentent une forte proximité avec un prototype selon une dimension, mais une distance significative selon l'autre dimension.
  
- Élaboration d'une solution méthodologique : conception d'un mécanisme basé sur un seuil d'acceptation global permettant d'évaluer objectivement si un vecteur peut être intégré à un prototype existant, en tenant compte du nombre de bit qui le sépare du prototype et en faisant la somme de ces bits de différence.

**Samy**

- Implémentation du code pour la visualisation graphique de l'aglgorithme MLD:
  - Utilisation de la bibliothèque matplotlib pour représenter graphiquement l'algorithme de clustering
  - Pour chacun des clusters on choisit une couleur pour différencier les différents clusters représentés sur une droite numérique

## 16/04/2025

**Anas et Issa**

- **Révision fondamentale de l'approche de clustering bidimensionnel** : Nous avons reconsidéré notre stratégie initiale qui consistait à évaluer la proximité d'un vecteur selon ses composantes x et y distinctement. Cette approche s'est révélée problématique car les dimensions ne peuvent être isolées de manière transparente. Notre nouvelle approche consiste à comparer directement le vecteur arrivant avec les prototypes existants dans leur représentation binaire intégrale, en évaluant les similarités et différences bit à bit. Cette comparaison holistique permet de déterminer objectivement si le vecteur doit être rattaché à un cluster existant ou initier un nouveau groupe.

- **Élaboration du cadre évaluatif pour la méthode unidimensionnelle** : il serait interessant de développer un protocole de validation comparative entre notre méthode de contraste et les algorithmes de clustering traditionnels comme K-means. Ce cadre permettra d'évaluer la robustesse face au bruit et la pertinence sur un nombre de dimensions importantes. Notre méthode MDL (Minimum Description Length) unidimensionnelle s'appuie sur trois critères complémentaires :
  1. **Hiérarchie structurelle** : Distance topologique dans l'arbre entre prototype et instance, quantifiée par le nombre de bits modifiés
  2. **Similarité récente** : Identification d'instances similaires récemment traitées
  3. **Popularité prototype** : Préférence pour les prototypes associés à de nombreuses instances

  L'algorithme calcule le coût MDL selon ces trois critères et sélectionne le prototype offrant la valeur minimale, optimisant ainsi la représentation des données.

  On peut par ailleurs noté qu'il y aura aussi un critère de complexité qui sera à prendre en compte pour le choix du prototype. En effet, si un prototype est trop complexe , on pourra mettre une limite sur le nombre de d'instance qu'il peut contenir et de ce fait venir plutôt accrocher cette instance à un niveau plus haut dans l'arbre.

## 05/05/2025

**Anas, Hussein et Samy**

- Continuation de l'avancée sur le travail de clustering en 1D(notamment avec MDL):
  - Tests avec une génération de données : résultats peu satisfaisant car l'algorithme donne trop de clusters à chaque
  - Avancée sur le code pour l'algorithme
  - Discussion sur la manière d'utiliser l'agorithme MDL au sein du projet
  - Deuxième avec des lois gaussiennes centrées en différents points : l'algorithme donne encore une fois trop de clusters.

**Issa**

- Travail sur MDL, recherche bibliographie

## 08/05/2025

**Hussein**

- Avancé sur l'alogrithme de clustering MDL:
  - Finalisation de l'écriture de l'algorithme
  - Tests sur des jeux de données générés aléatoirement
  - Réussite global mais le code a tendance à donner trop de clusters

## 10/05/2025

**Samy**

- Travail sur la visualisation de l'algorithme de clustering en 1D:
  - Création d'image avec `matplotlib` pour chaque étape lors de l'ajout incrémental
  - Utilisation de `ffmpeg` pour générer une vidéo à partir de chacune des images des différentes étapes

## 11/05/2025

**Hussein**

Création de vidéo à l'aide de ffmpeg, pour visualiser le clustering incrémental


## 12/05/2025

**Anas, Issa, Hussein et Samy**

- Discussion avec l'encadrant et tous les membres de l'avancée du projet:
  - Explication de l'avancé sur la 1D avec algorithme finalisé
  - Étude de la visualisation d'un exemple en 1D : trop de clusters sont créés 
  - Discussion sur la direction à prendre et la suite pour la dimension 1
  - Discussion sur la manière d'implémenter les dimensions supérieurs à partir de la première

## 14/05/2025

**Toute l'équipe**

- Rédaction du rapprot sur les enjeux environnementaux et sociaux du projet:
  - Définition des différents enjeux
  - Discussion sur l'apport de notre projet à ces différentes enjeux : réduction de biais, confidentialité, consommation d'énergie
  - Rédaction du rapport final et envoi

## 18/05/2025

**Anas, Issa, Hussein et Samy**

- Préparation de la présentation devant les autres groupes du 19/05/2025
  - Travail sur l'explication du projet
  - Mise en évidences des problématiques et difficultés : pas le droit à beaucoup de données, recherche
  - Mise en avant des points forts du projet : consommation, universalité, perte du besoin d'une distance, élimination des biais
  - Élaboration d'un plan de présentation

## 19/05/2025

**Anas, Issa, Hussein et Samy**

- Passage devant deux jurys et présentaton du projet pour avoir des retours :
  - Globalement les retours sont positifs
  - Explication des enjeux et des aspects techniques à revoir
  - L'ambition du projet semble raisonnables aux jurys d'élèves

## 28/05/2025
**Anas**

Implémentation d'un arbre binaire récursif pour le stockage de nombres binaires

- Développement de la structure avec navigation basée sur les bits de différence (0→gauche, 1→droite)
- Création des algorithmes d'insertion et de suppression avec gestion des nœuds indéterminés
- Implémentation d'un module de visualisation graphique avec matplotlib
- Documentation de 6 approches alternatives : dataclasses, pattern Strategy, générateurs, cache, vectorisation NumPy, thread-safety
- Tests sur l'insertion de valeurs 4 bits : [8, 0, 12, 10, 4, 14, 2, 6]



## 31/05/2025 

**Issa**

Implémentation de l'algorithme de clustering MDL pour les points 2D

- Utilisation du principe de Minimum Description Length (MDL) pour déterminer si un point doit rejoindre un cluster existant ou en créer un nouveau.
- Ajout d'un paramètre de seuil

## Du 09/06/2025 au 13/06/2025

**Samy et Hussein**

- Fintion de l'algortihme MDL en 1D:
  - Tests sur différents sets de données
  - Résultats satisfaisants dans la globalité
  - Toujours des problèmes et inprécisions sur des points particuliers

## 10/06/2025

**Issa et Hussein**

-Rencontre avec l'encadrant
  - Discussion sur la 2D
  - Discussion autour du rapport final et de la présentation

## 21/06/2025

**Samy**

- Reprise de la visualisation de la simulation :
  - Vidéos données par ffmpeg trop rapides et on ne comprend pas correctement ce qui est montré
  - Changement pour dessiner un arbre à chaque étape qui va de la racine aux feuilles qui sont les données 
  - Utilisation  de `networkx`

## 23/06/2025
**Tous le groupe**
- Mis à jour du planning pour la finalisation du projet 
- Rediscussion sur tableau de la théorie que l'on implémentera en 2D

**Hussein**

- Finalisation de l'algorithme en 1D et ajout d'un paramètre : l'acuité. Plus l'acuité est petite plus on zoom sur les clusters ie les clusters sont de plus en plus spécifique, plus l'acuité est grande plus les clusters sont généraux

**Anas**

Implémentation d'un arbre hiérarchique binaire pour le clustering 2D (fichier tree.py) :
- Classe `BinaryHierarchicalTree` avec insertion automatique et création de prototypes généralisés
- Algorithme de subsumption utilisant des wildcards (*) pour représenter les patterns communs
- Système de visualisation avec NetworkX pour afficher la structure arborescente
- Tests de validation avec données spécifiques et aléatoires pour vérifier la robustesse

**Issa**

Implémentation d'un arbre hiérarchique pour le clustering 3D

## 24/06/2025

**Hussein**

- Mise en forme en papier de la démarche suivi en 1D. Premier essai de généralisation à la 2D en suivant la méthode du professeur 
- La 2D par entrlacement des bits de chacune des dimensions pour former un mot en 1D et appliquer ce qui a été fait en 1D est une mauvaise piste

**Samy**

- Travail le code d'un algorithme en 2D :
  - Le principe est de rélaiser un clustering 1D sur chqque dimension pour ensuite assigner au cluster le plus proche en 2D. Si aucun cluster ne correspond il faut créer un nouveau.
  - Difficulté avec la condition de création d'un nouveau cluster
  - Problème avec le mélange des deux dimensions.

**Issa**

- Travail sur l'arbre hiérarchique pour le clustering 3D

  - Représentation des motifs binaires en tuples de chaînes.
  - Création dynamique de l'arbre à partir des données d'entrée.
  - Détection et gestion des relations de subsomption.
  - Création automatique de prototypes si aucun nœud existant ne peut intégrer un nouveau motif.

**Anas**

Implémentation complète de l'algorithme MDL pour l'arbre hiérarchique 2D :
- Calcul automatique des scores MDL basé sur rang (profondeur) + popularité (nombre de données directes)
- Élagage intelligent des prototypes selon le principe du minimum de description : garde les prototypes avec scores minimaux
- Sélection du prototype optimal par minimisation de la somme popularité + rang parmi les candidats généralisant les données
- Visualisation des scores MDL et réorganisation automatique de l'arbre après élagage

## 25/06/2025

**Anas, Issa, Hussein, Samy**

- Finalisation du travail:
  - Visualtion sur toutes les approches : 1D, 2D et ND
  - Écriture finale de tous les codes pour toutes les dimensions
  - Relecture des articles pour préparation de la bibliographie
  - Préparation du compte rendu


## 26/05/2025

**Anas, Issa, Hussein, Samy**

- Préparation de la présentation pour l'encadrant:
  - Retour sur l'ensemble de l'année et des approches utilisées
  - Préparation des slides de présentation
  - Explication des différentes méthode et de l'approche de recherceh suivi

- Présentation devant l'encadrant et feedback, retour de l'encadrant sur ce qui pourrait être amélioré:
  - Insister sur les avantages
  - Bien présenter les problématiques avant les solutions
  - Insister sur la simplicité




