import math
import numpy as np
import random

# Classe Prototype pour représenter un prototype de cluster
class Prototype:
    def __init__(self, motif):
        self.motif = motif          # motif binaire généralisé (chaîne de '0','1','*')
        self.parent = None          # lien vers le prototype parent (plus général)
        self.enfants = []           # liste des prototypes enfants (plus spécifiques)
        self.instances = []         # liste des instances binaires appartenant à ce prototype

    def longueur_prefixe(self):
        # Calculer la longueur du préfixe spécifique (nombre de bits avant le premier joker '*')
        return self.motif.index('*') if '*' in self.motif else len(self.motif)

# Fonction pour vérifier si le motif A couvre le motif B (A plus général ou égal à B)
def motif_couvre(motifA, motifB):
    # On suppose que les deux motifs ont la même longueur
    assert len(motifA) == len(motifB)
    for bitA, bitB in zip(motifA, motifB):
        if bitB == '*':
            # Si B a un joker à cette position mais A a un bit spécifique,
            # alors A ne couvre pas toutes les possibilités que B couvre.
            if bitA != '*':
                return False
        else:
            # Si B a un bit spécifique
            if bitA != '*' and bitA != bitB:
                # A a un bit différent (0/1) de B -> ne couvre pas B
                return False
            # (Si bitA == '*' ou bitA == bitB, c'est acceptable pour cette position)
    return True

# Fonction pour calculer le coût MDL (longueur description modèle + données) pour une liste de prototypes
def calculer_cout_MDL(prototypes):
    # Ne considérer que les prototypes qui possèdent au moins une instance (clusters non vides)
    clusters = [proto for proto in prototypes if len(proto.instances) > 0]
    # Coût du modèle : somme des longueurs de motifs (bits spécifiés) sans double-compter les préfixes hérités
    cout_modele = 0
    for proto in clusters:
        if proto.parent is None:
            # Ajouter tous les bits spécifiés du prototype (jusqu'au premier '*')
            cout_modele += proto.longueur_prefixe()
        else:
            # Ajouter seulement les bits spécifiés au-delà de ceux du parent (préfixe propre)
            extra_bits = proto.longueur_prefixe() - proto.parent.longueur_prefixe()
            if extra_bits < 0:
                extra_bits = 0
            cout_modele += extra_bits
    # Coût des données : somme des coûts d'encodage de chaque instance étant donné les prototypes
    M = len(clusters)
    # Coût pour coder l'identifiant du cluster (log2(M) bits par instance, nul si M=1)
    cout_id_cluster = math.log2(M) if M > 1 else 0.0
    cout_donnees = 0.0
    for proto in clusters:
        # Coût d'assignation des instances de ce cluster (identifiant du cluster pour chaque instance)
        cout_donnees += cout_id_cluster * len(proto.instances)
        # Coût des bits à encoder pour les positions en joker dans le motif du prototype
        total_bits = len(proto.motif)
        prefixe = proto.longueur_prefixe()
        nb_jokers = total_bits - prefixe
        # Chaque instance fournit nb_jokers bits propres (non couverts par le motif) à encoder
        cout_donnees += nb_jokers * len(proto.instances)
    return cout_modele + cout_donnees

# Fonction pour mettre à jour les liens parent-enfant entre prototypes (structure hiérarchique)
def mettre_a_jour_liens_parents(prototypes):
    # Réinitialiser les liens de parenté
    for proto in prototypes:
        proto.parent = None
        proto.enfants = []
    # Trier les prototypes par longueur de préfixe décroissante (les plus spécifiques d'abord)
    protos_tries = sorted(prototypes, key=lambda p: p.longueur_prefixe(), reverse=True)
    # Assigner le parent le plus général (au préfixe le plus court) couvrant chaque prototype
    for i, enfant in enumerate(protos_tries):
        for j in range(i+1, len(protos_tries)):
            parent = protos_tries[j]
            if motif_couvre(parent.motif, enfant.motif):
                # Associer comme parent le prototype plus général `parent` (remplace un parent plus général s'il existait)
                if enfant.parent is None or parent.longueur_prefixe() > enfant.parent.longueur_prefixe():
                    enfant.parent = parent
    # Mettre à jour les listes d'enfants de chaque parent
    for proto in prototypes:
        if proto.parent is not None:
            proto.parent.enfants.append(proto)

# Fonction principale pour insérer une nouvelle instance dans le modèle de prototypes existant
def inserer_instance(instance, prototypes):
    # 3. Calculer le coût si on crée un nouveau prototype (cluster) pour cette instance
    nouveau_proto = Prototype(instance)
    nouveau_proto.instances.append(instance)
    cout_nouveau = calculer_cout_MDL(prototypes + [nouveau_proto])
    meilleur_cout = cout_nouveau
    choix = "nouveau"
    prototype_selectionne = None
    motif_generalise = None

    # Parcourir chaque prototype existant pour tester le rattachement de l'instance
    for proto in prototypes:
        # Sauvegarder l'état initial du prototype
        motif_orig = proto.motif
        parent_orig = proto.parent
        enfants_orig = proto.enfants[:]
        instances_orig = proto.instances[:]

        # 1. Identifier le premier bit de gauche où le prototype diffère de l'instance
        proto.instances.append(instance)  # ajout temporaire de l'instance
        k = None
        for idx, bit in enumerate(motif_orig):
            if bit == '*':
                continue  # le joker correspond à n'importe quelle valeur, pas de différence ici
            if idx >= len(instance):
                break  # par précaution, si instance plus courte (longueur fixe supposée égale)
            if motif_orig[idx] != instance[idx]:
                k = idx
                break

        # 2. Généraliser tous les bits à partir de cette position k en jokers '*'
        nouveau_motif = motif_orig
        if k is not None:
            nouveau_motif = motif_orig[:k] + "*" * (len(motif_orig) - k)
        proto.motif = nouveau_motif

        # Si le parent du prototype ne couvre plus le nouveau motif généralisé, on coupe le lien parent-enfant (temporairement)
        if proto.parent is not None and not motif_couvre(proto.parent.motif, proto.motif):
            proto.parent = None

        # 3. Calculer le coût MDL (modèle + données) pour ce rattachement hypothétique
        cout_rattachement = calculer_cout_MDL(prototypes)
        # 4. Mémoriser l'option la meilleure (coût minimal) entre rattachement et création d'un nouveau prototype
        if cout_rattachement < meilleur_cout:
            meilleur_cout = cout_rattachement
            choix = "rattachement"
            prototype_selectionne = proto
            motif_generalise = nouveau_motif

        # Restaurer l'état initial du prototype (annuler les modifications temporaires)
        proto.motif = motif_orig
        proto.parent = parent_orig
        proto.enfants = enfants_orig
        proto.instances = instances_orig

    # Appliquer l'option retenue avec mise à jour du modèle de clusters
    if choix == "nouveau":
        # Créer un nouveau prototype indépendant pour l'instance entrante
        prototypes.append(nouveau_proto)
        prototype_selectionne = nouveau_proto
        motif_generalise = nouveau_proto.motif
    else:
        # Rattacher l'instance au prototype existant choisi et généraliser son motif
        prototype_selectionne.motif = motif_generalise
        prototype_selectionne.instances.append(instance)

    # 5. Mettre à jour la structure hiérarchique des prototypes (liens parent-enfant)
    mettre_a_jour_liens_parents(prototypes)
    return choix, prototype_selectionne

# --- Exemple d'utilisation ---
prototypes = []

def generate_binary_numbers(num_bits, num_numbers):
    binary_numbers = []
    for _ in range(num_numbers):
        
        binary_number = ''.join(random.choice('01') for _ in range(num_bits))
        binary_numbers.append(binary_number)
    return binary_numbers


num_bits = 4
num_numbers = 100  

donnees = generate_binary_numbers(num_bits, num_numbers)



for inst in donnees:
    decision, proto = inserer_instance(inst, prototypes)
    print(f"Instance {inst} -> {decision} au prototype {proto.motif}")
