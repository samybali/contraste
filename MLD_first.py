"""
Telecom Paris : projet Contraste

Projet réalisé par : Samy Bali, Hussein El gouch, Issa Daou, Anas El Barhdadi

Encadré par Jean-Louis Dessalles
"""


class SGGGraph:
    """
    Graphe hiérarchique de prototypes pour clustering incrémental.
    Chaque prototype est un motif binaire généralisé (avec '0', '1' ou '*' par position).
    """
    
    class Prototype:
        """Représentation d’un prototype (motif généralisé) dans le graphe."""
        def __init__(self, pattern:str):
            self.pattern = pattern        # Motif binaire du prototype (chaîne de '0','1','*')
            self.count = 1               # Nombre d'instances associées à ce prototype
            self.parent = None           # Prototype parent (plus général) dans le graphe
            self.children = []           # Liste des prototypes enfants (plus spécifiques)
    
    def __init__(self, bit_length:int=None):
        """
        Initialise le graphe SGG. 
        :param bit_length: longueur binaire fixe des instances (optionnel, peut être déterminée dynamiquement).
        """
        self.prototypes = []            # Liste des prototypes existants (instances de SGGGraph.Prototype)
        self.bit_length = bit_length    # Longueur en bits des instances/prototypes (si None, déterminée par la première instance)
    
    def _to_binary(self, instance:int) -> str:
        """
        Convertit un entier en binaire (sans signe) avec la longueur requise.
        Utilise la longueur fixée du graphe, ou ajuste cette longueur si l'instance nécessite plus de bits.
        """
        # Conversion en binaire sans le préfixe '0b'
        binary_str = bin(instance)[2:]
        if self.bit_length is None:
            # Première instance : on fixe la longueur sur la taille binaire minimale de cette instance
            self.bit_length = len(binary_str)
        if len(binary_str) > self.bit_length:
            # Si l'instance dépasse la longueur actuelle, on étend tous les prototypes existants
            extra_bits = len(binary_str) - self.bit_length
            for proto in self.prototypes:
                # On ajoute des '0' en tête des motifs existants (toutes leurs instances avaient ces bits à 0)
                proto.pattern = '0'*extra_bits + proto.pattern
            self.bit_length = len(binary_str)
        # On pad à gauche avec '0' jusqu'à la longueur fixée
        return binary_str.zfill(self.bit_length)
    
    def assign_instance(self, instance:int):
        """
        Assigne une nouvelle instance (entier) à un prototype selon le principe MDL.
        Crée ou généralise un prototype en conséquence, et met à jour le graphe hiérarchique.
        """
        # 1. Représentation binaire de l'instance (avec la longueur adéquate)
        instance_bits = self._to_binary(instance)
        L = len(instance_bits)  # longueur binaire de l'instance (et des prototypes à ce stade)
        
        # Si aucun prototype n'existe encore, on crée directement un prototype pour cette instance
        if not self.prototypes:
            new_proto = SGGGraph.Prototype(instance_bits)
            self.prototypes.append(new_proto)
            # Pas de parent à attribuer (c'est le premier prototype, le plus général pour son motif)
            return new_proto  # On retourne éventuellement le prototype créé
        
        # Initialisation du meilleur coût (coût d'un nouveau prototype séparé) et du candidat de prototype choisi
        best_cost = L  # coût MDL si on crée un nouveau prototype (motif de L bits spécifiques, W=0)
        best_proto = None
        best_new_pattern = None  # motif généralisé du prototype si on l'utilise
        
        # 2. Calcul du coût MDL pour rattacher l'instance à chaque prototype existant
        for proto in self.prototypes:
            pattern = proto.pattern
            # Vérification des différences sur bits spécifiques
            conflicts = []  # indices des bits où le prototype a une valeur spécifique différente de l'instance
            for i, bit in enumerate(pattern):
                if bit != '*' and bit != instance_bits[i]:
                    conflicts.append(i)
            d = len(conflicts)          # nombre de bits à généraliser si on utilise ce prototype
            # Si le prototype ne peut pas couvrir l'instance sans différence (d > 0),
            # on évalue la généralisation potentielle en créant un nouveau motif généralisé
            if d > 0:
                # Construire le nouveau motif généralisé en remplaçant les bits en conflit par '*'
                pattern_list = list(pattern)
                for idx in conflicts:
                    pattern_list[idx] = '*'
                new_pattern = ''.join(pattern_list)
            else:
                # Aucune différence sur les bits spécifiques : l'instance est déjà couverte par le motif tel quel
                new_pattern = pattern
            
            # Calcul du coût total (MDL) si on rattachait l'instance à ce prototype
            S_old = pattern.count('0') + pattern.count('1')    # bits spécifiques actuels
            W_old = pattern.count('*')                         # jokers actuels
            N_old = proto.count                                # nombre d'instances actuelles dans ce prototype
            # Nouveau nombre de jokers et de spécifiques après généralisation (si d > 0)
            S_new = new_pattern.count('0') + new_pattern.count('1')
            W_new = new_pattern.count('*')
            # Coût additionnel en bits pour rattacher l'instance à ce prototype (Delta MDL)
            # Formule : Delta L = N_old * d + W_old  (simplification du calcul MDL local)
            cost_increase = N_old * d + W_old
            # Comparaison avec le meilleur coût trouvé
            if cost_increase < best_cost:
                best_cost = cost_increase
                best_proto = proto
                best_new_pattern = new_pattern
        
        # 3. Décision : rattacher à best_proto si avantageux, sinon créer un nouveau prototype
        if best_proto is not None and best_cost < L:
            # Rattachement à un prototype existant
            # Mettre à jour le prototype choisi (généralisation + compteur)
            if best_new_pattern is not None:
                # Si le motif change (généralisation de nouveaux jokers), on met à jour le pattern du prototype
                if best_new_pattern != best_proto.pattern:
                    best_proto.pattern = best_new_pattern
            best_proto.count += 1  # l'instance est désormais couverte par ce prototype
            updated_proto = best_proto
        else:
            # Aucun prototype n'offre un gain suffisant, création d'un nouveau prototype spécifique
            updated_proto = SGGGraph.Prototype(instance_bits)
            self.prototypes.append(updated_proto)
        
        # 4. Mise à jour du graphe hiérarchique des prototypes (parent/enfant)
        # Cas 4a : Si un prototype a été généralisé (updated_proto était existant)
        if best_proto is not None and best_cost < L:
            # Vérifier si updated_proto couvre désormais d'autres prototypes qui pourraient devenir ses enfants
            for proto in self.prototypes:
                if proto is updated_proto:
                    continue
                # Un prototype A "couvre" un prototype B si pour chaque bit de A:
                # - si A a un bit spécifique, B a le même bit à cette position
                # - si A a '*', B peut avoir n'importe quoi à cette position
                # => Le motif A est plus général ou égal à B.
                can_cover = True
                for bit_a, bit_b in zip(updated_proto.pattern, proto.pattern):
                    if bit_a != '*' and bit_a != bit_b:
                        can_cover = False
                        break
                if can_cover:
                    # Le prototype 'updated_proto' couvre 'proto'
                    # Si proto avait déjà un parent moins spécifique (plus général) ou pas de parent, on le rattache
                    if proto.parent is None or proto.parent is not updated_proto:
                        # Retirer proto de l'ancien parent s'il en avait un
                        if proto.parent is not None:
                            try:
                                proto.parent.children.remove(proto)
                            except ValueError:
                                pass
                        proto.parent = updated_proto
                        if proto not in updated_proto.children:
                            updated_proto.children.append(proto)
        # Cas 4b : Si un nouveau prototype a été créé
        else:
            new_proto = updated_proto
            # Trouver un prototype parent existant qui le couvre (le plus spécifique possible)
            best_parent = None
            best_parent_specific_bits = -1
            for proto in self.prototypes:
                if proto is new_proto:
                    continue
                can_cover = True
                for bit_a, bit_b in zip(proto.pattern, new_proto.pattern):
                    if bit_a != '*' and bit_a != bit_b:
                        can_cover = False
                        break
                if can_cover:
                    # proto peut couvrir new_proto; on choisit le plus précis (max de bits spécifiques)
                    S_bits = proto.pattern.count('0') + proto.pattern.count('1')
                    if S_bits > best_parent_specific_bits:
                        best_parent_specific_bits = S_bits
                        best_parent = proto
            # Assigner le parent si trouvé
            if best_parent is not None:
                new_proto.parent = best_parent
                best_parent.children.append(new_proto)
        
        return updated_proto  # Renvoie le prototype auquel l'instance a été affectée (pour suivi éventuel)
