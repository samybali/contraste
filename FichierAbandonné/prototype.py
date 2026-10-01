"""
Telecom Paris : projet Contraste

Projet réalisé par : Samy Bali, Hussein El gouch, Issa Daou, Anas El Barhdadi

Encadré par Jean-Louis Dessalles

Ce fichier contient la définition de la classe qui permet de représenter les prototypes.

"""

class Prototype:
    """
    Classe permettant de représenter les prototypes de nos données
    
    """

    def __init__(self, nb_bits, nb_stars, core, bits_ignored):
        """
        Construction de la classe
        """
        self.nb_bits = nb_bits
        self.nb_stars = nb_stars
        self.core = core
        self.bits_igonred = bits_ignored

    def calcutate_pop(self, list):
        """
        Calcul de la popularité d'un prototype
        """

        pop = 0

        for elt.core in list:
            for i in range(self.nb_bits - self.nb_stars):
                if elt.core[i] != self.core[i]:
                    continue
            pop += 1

        return pop


    def find_descendant(self, list):
        """
        Permet de trouver les descendants d'un prototype parmi une liste
        """
        desc = []

        for elt in list:
            for i in range(self.nb_bits - self.nb_stars):
                if elt.core[i] != self.core[i]:
                    continue
            desc.append(elt)

        return desc

