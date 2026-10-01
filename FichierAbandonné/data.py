"""
Telecom Paris : projet Contraste

Projet réalisé par : Samy Bali, Hussein El gouch, Issa Daou, Anas El Barhdadi

Encadré par Jean-Louis Desssalles

Ce fichier contient la définition de la classe qui permet de représenter les données que l'on souhaite clusteriser.

"""

import prototype
import numpy as np

NB_BITS = 20 # Nombre de bits utilisés pour la représentation des données en binaire

BITS_IGNORED = 3 # Nombre de de bits qui ne sont pas pris en compte

class Data:
    """
    Classe représentant les données traitées

    Attributs :
    - nb (int): Donnée
    - nb_bits (int): Le nombre de bits utilisés pour la représentation en binaire
    - bin_rep (str): Représentation binaire du nombre
    - bits_ignored (int): Nombre de bits qui ne sont pas pris en compte
    - prototype (Prototype): Prototype associé à la donnée considérée
    - nearest (Data): Donnée la plus proche

    """
    def __init__(self, nb, nb_bits, bits_ignored):
        """
        Constructeur de la classe
        """
        self.nb = nb
        self.nb_bits = nb_bits
        self.bin_rep = bin(nb)
        self.bits_ignored = bits_ignored
        self.prototype = None
        self.nearest = None

    def find_nearest(self, list):
        """
        Cette méthode trouve l'élement le plus proche dans la liste donnée en argument
        """
        list_bis = list
        list_bis.remove(self.nb)

        nearest = min(list_bis, key=lambda x: abs(x - self.nb))

        return Data(nearest, NB_BITS, BITS_IGNORED)
    
        

