"""
Telecom Paris : projet Contraste

Projet réalisé par : Samy Bali, Hussein El gouch, Issa Daou, Anas El Barhdadi

Encadré par Jean-Louis Desssalles

Ce fichier contient la définition de la classe qui permet de représenter les liens entre les différents prototypes sous forme d'arbre

"""

import prototype

class Tree:
    """
    Classe arbre des liens entre les différents prototypes

    Attributs :
    - left (Tree): sous-arbre gauche
    - right (Tree): sous-arbre droit
    - proto (Tree): Prototype du noeud

    """

    def __init__(self, proto):
        """
        Constructeur de la classe
        """
        
        self.left = None
        self.right = None
        self.proto = proto

    def find_nearest(self, target):
        """
        Méthode qui trouve le prototype le plus similaire à une cible donnée
        
        Args:
            target: Le prototype ou vecteur de caractéristiques cible
            
        Returns:
            Le prototype le plus similaire dans l'arbre
        """
        # Comparer avec le nœud actuel
        best_proto = self.proto
        best_similarity = self.proto.similarity(target)
        
        # Chercher dans le sous-arbre gauche s'il existe
        if self.left is not None:
            left_proto = self.left.find_nearest(target)
            left_similarity = left_proto.similarity(target)
            if left_similarity > best_similarity:
                best_similarity = left_similarity
                best_proto = left_proto
        
        # Chercher dans le sous-arbre droit s'il existe
        if self.right is not None:
            right_proto = self.right.find_nearest(target)
            right_similarity = right_proto.similarity(target)
            if right_similarity > best_similarity:
                best_similarity = right_similarity
                best_proto = right_proto
        
        return best_proto