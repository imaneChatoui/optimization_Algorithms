
from numpy import *  # importe tout de numpy sans préfixer


class Simplex(object):  # Classe Simplex utilisée pour résumer et décomposer l'algorithme en diverses méthodes

    def __init__(self, obj):
        # __init__ est un constructeur, qui initialise une instance d'une classe ; entrée : fonction objectif ;
        #          # sortie: instance initialisée d'une classe

        self.obj = [1] + obj  # liste incl. fonction objectif; 1 correspond à une valeur z égale à 1 dans la ligne 0 de augm. matr.
        self.rows = []  # liste vide avec lhs de contraintes pour la forme matricielle augmentée
        self.cons = []  # liste vide pour les valeurs rhs des contraintes
        self.decvar = obj  # !stocke les variables de décision dans la fonction objectif sous forme de liste

    def add_constraint(self, lhsexpression, rhsvalue):
        # used pour ajouter une contrainte; entrée : lhs et rhs de contrainte sous la forme générale x+y <= c -> [x, y] <= [rhsvalue];
        #          # add_constraints([lhs], rhsvalue); sortie: stocke la contrainte dans

        self.rows.append([0] + lhsexpression)  # 0 représente la colonne de valeur z, lhsexpression pour lhs de contraintes
        self.cons.append(rhsvalue)  # !la valeur est égale à la valeur rhs de la contrainte

    def pivot_column(self):
        # identifie l'index de la colonne pivot input (pas en tant que paramètre, mais dans la boucle for de la méthode):
        #          # fonction objectif (c'est-à-dire ligne 0); sortie: index de la colonne pivot

        low, idx = 0, 0  # !deux variables sont utilisées pour voir quelles valeurs de la ligne 0 sont non négatives
        for i in range(1, len(self.obj)-1):
            # boucle sur les variables de décision de la ligne 0 ; -1 car la liste commence à l'index 0
            if self.obj[i] < low:  # conditionnel qui est vrai pour les entrées négatives de la ligne 0
                low = self.obj[i]  # définit bas sur la valeur de coefficient négatif la plus faible
                idx = i  # sets idx à l'index de la variable entrante (EV), c'est-à-dire le plus grand coefficient négatif de la ligne 0
        if idx == 0:
            return -1  # if idx remains 0, i.e. no negative value in row 0, -1 is returned
        return idx

    def pivot_row(self, col):
        # identifie l'index de la ligne pivot ; entrée : colonne pivot ; sortie : plus petite valeur de rhs/evc

        rhs = [self.rows[i][-1] for i in range(len(self.rows))]
        # crée la liste de la colonne la plus à droite dans la matrice augmentée

        lhs = [self.rows[i][col] for i in range(len(self.rows))]
        #crée une liste en appelant récursivement des éléments dans les lignes sauf la ligne 0 dans la colonne pivot, col

        ratio = []  # eliste vide pour les ratios de (RHS de la ligne i)/(entrée du coefficient de valeur de la ligne i)
        for i in range(len(rhs)):  # commence par zéro jusqu'à, mais non compris, le nombre d'éléments dans la liste de droite
            if lhs[i] == 0:  # conditionnel pour éviter une ZeroDivisionError
                ratio.append(99999999 * abs(max(rhs)))  # ajoute un nombre énorme afin de s'assurer de ne pas le choisir
                continue
            ratio.append(rhs[i]/lhs[i]) # ajoute des valeurs à la colonne RHS/EVC
        try: lowestratio = min(i for i in ratio if i > 0)  # !trouve le rapport positif le plus faible
        except: raise ValueError("RHS/EVC ratios are all negative. Try a different example.")  # cas très très rare
        return ratio.index(lowestratio)  # !renvoie l'indice de la plus petite valeur positive de RHS/EVC

    def pivot(self, row, col):  # Opérations de ligne Gauss-Jordan autour du point de pivot; entrée: index de ligne et de colonne vers
                                # spécifiez le point de pivot; sortie: modifie l'instance de classe (c'est-à-dire la fonction objectif et
                                #lignes dans la matrice augmentée)
        pivotpoint = self.rows[row][col]  # !définissez le point de pivot
        self.rows[row] /= pivotpoint  # changer le point de pivot à 1 en divisant la ligne respective par la valeur du point de pivot
        self.obj = self.obj - self.obj[col]*self.rows[row]  # définit l'entrée au-dessus du point de pivot dans la ligne 0 égale à 0
        for r in range(len(self.rows)):  # boucle à travers les entrées de la matrice augmentée au-dessus et au-dessous de la ligne pivot
            if r == row:
                continue  # saute la ligne de pivot
            self.rows[r] = self.rows[r] - self.rows[r][col]*self.rows[row]
            # définit les entrées sous le point de pivot égal à 0, sauf pour la ligne 0 elle-même

    def solutionvalues(self): # méthode entièrement auto-codée. imprimer les valeurs x de la solution optimale, entrée: toutes les contraintes,
        # sortie: liste des valeurs x au point optimal

        xlist = [0] * len(self.decvar) # un tableau des zero taille de tableau des variable de desicion
        # liste de 0 avec une longueur égale au nombre de variables de décision pour éventuellement produire des valeurs de solution

        for i in range(len(self.rows)):  # boucle pour itérer sur chaque ligne sauf la ligne 0
            for x in range(len(self.rows[i])):  #boucle pour itérer sur chaque colonne de self.rows[i]
                while x < len(xlist):  # le 1 ne peut être que dans les colonnes (nombre de variables de décision)+1
                    if self.rows[i][x+1] == 1 and self.is_unit_column(x+1):
                        # si l'entrée est égale à 1 ET la colonne est une colonne unitaire

                        xlist[x] = self.rows[i][-1]  # si tel est le cas, il ajoute le dernier élément de ligne à xlist
                    x += 1  # fait que la boucle while inspecte l'élément suivant dans la ligne i
                break  # coupe la deuxième boucle for pour revenir à la première boucle for et augmente i de 1
        return xlist

    def is_unit_column(self, col):  # eméthode entièrement auto-codée.
        # vérifie si une colonne spécifique est une colonne d'unité; entrée: index de colonne; sortie: booléen Vrai ou Faux

        row_iterator = 0  # variable pour parcourir les lignes dans self.rows
        columnsum = 0 + self.obj[col]  # somme égale à 1 si c'est une colonne d'unité
        while row_iterator < len(self.cons):  # boucle sur le nombre de contraintes, c'est-à-dire le nombre de lignes à vérifier
            columnsum += self.rows[row_iterator][col]  # calcule la somme de chaque élément de la colonne col
            row_iterator += 1  # 1 est ajouté à la variable d'itérateur pour vérifier la ligne suivante
        if columnsum == 1:  # conditionnel qui renvoie True si la somme des entrées de la colonne est égale à 1
            return True
        else:  # conditionnel qui renvoie False si la somme des entrées de colonne est égale à une valeur différente de 1
            return False

    def row0_check(self):
        # vérifie si les coefficients de décision et les variables d'écart de la ligne 0 sont tous non négatifs; sorties opérateur booléen
        if min(self.obj[1:-1]) >= 0:
            return 1
        # vérifie toutes les entrées sauf la toute première, qui doit de toute façon être égale à 1; 1 signifie booléen vrai

        else:
            return 0  # renvoie un 0 (c'est-à-dire booléen Faux) pour réitérer car les solutions optimales n'ont pas encore été trouvées

    def display(self):
        # imprime d'abord la matrice augmentée de la fonction objectif et des contraintes, puis chaque itération de la matrice augmentée
        # entrée: données stockées dans l'instance de classe; sortie: représentation matricielle et valeurs z à chaque itération

        print('\nMatrix form:\n', matrix([self.obj] + self.rows), '\n\n', 'z = {}'.format(self.obj[-1]))
        """z=self.obj[-1]
        matrix([self.obj] + self.rows)"""
        # !imprime la matrice et la valeur z à chaque itération

    def solve(self):
        # méthode pour résoudre le problème LP; entrée : soi. données dans l'instance de classe ; sortie : imprimer les itérations simplex et la solution

        iterationCounter = 0

        # construire un tableau matriciel augmenté
        for rowindex in range(len(self.rows)): # !boucle pour stocker les lignes 1 à i, y compris les variables d'écart
            self.obj += [0]  # ajoute des zéros à la ligne 0 pour avoir la même longueur que les lignes 1 à i

            slackvar = [0 for _ in range(len(self.rows))]
            # !crée une liste de 0 avec une longueur égale au nombre de contraintes, égalant le nombre de variables d'écart

            slackvar[rowindex] = 1 # !remplacer les termes variables d'écart diagonal par 1

            self.rows[rowindex] += slackvar + [self.cons[rowindex]]
            # !crée des lignes complètes 1 à i en ajoutant des variables d'écart et la valeur rhs des contraintes

            self.rows[rowindex] = array(self.rows[rowindex], dtype=float)
            #  ! Définissez les lignes 1 à i dans un tableau (du package numpy) comprenant des contraintes, des variables d'écart et
            # rhs de contraintes sous forme de flottants

        self.obj = array(self.obj + [0], dtype=float)
        # définir la ligne 0 sur un tableau comprenant des 0 supplémentaires car il n'y a pas de self.cons (c'est-à-dire la valeur rhs)
        # pour la ligne 0, avec toutes les entrées sous forme de flottants
        #self.display()
        # imprime la matrice augmentée initiale avec tous les coefficients Les variables
        # et slack étant non négatives en appelant la méthode .display() spécifiée ci-dessus

        while not self.row0_check(): # répéter la méthode (c'est-à-dire que la méthode self.check renvoie 0)
            iterationCounter += 1  # compte les itérations

            pivotcol = self.pivot_column()
            # !pivotalcol  est égal à l'index de la colonne pivot moins 1 (les listes python commencent à 0)

            pivotrow = self.pivot_row(pivotcol)
            # !pivotrow est égal à la ligne pivot à la colonne pivotcol moins 2 (les listes python commencent à zéro et
            # La méthode self.pivot_row() n'inclut pas la ligne 0
            self.pivot(pivotrow, pivotcol)  # appelle la méthode .pivot(), pour la ligne r et la colonne c

            #print('\niteration: {}\npivot column: {}\npivot row: {}'.format(iterationCounter, pivotcol+1, pivotrow+2))
            #c+1 et r+2 pour compenser la liste python commençant à 0 et la ligne 0 n'est pas incluse dans self.pivot_row()
            # méthode respectivement

            # self.display()  # imprime la matrice augmentée avec tous les coefficients

        if self.row0_check(): # !imprime les valeurs x de la solution optimale une fois la dernière itération effectuée
            print("at x =", self.solutionvalues(), "\n--> my result")
            print( 'z = {}'.format(self.obj[-1]))
            #return self.solutionvalues()
"""t = Simplex([-2,-3,-2])
t.add_constraint([2, 1, 1], 4)
t.add_constraint([1, 2, 1], 7)
t.add_constraint([0, 0, 1], 5)
t.solve()"""

t=Simplex([-3,-2])
t.add_constraint([2,1],18)
t.add_constraint([2,3],42)
t.add_constraint([3,1],24)
t.solve()

"""
""""""
ob=[]
y=int(input("enter le nombre des variable "))
for i in range(y):
    ob.append(input("enter les coefficients de la fonction  objectif<< vous devez les entrer avec  l'inverce de ses signes>>:: "))
t=Simplex(ob)
x=int(input("enter le nombre de contrainte:"))
tab=[]
for i in range(x):
    print("contarint num:", i)
    srd = input("enter la valeur de second menbre de la contrainte:")
    for j in range(y):
        s=input("entrer les coieficients de la const:")
        tab.append(s)

    t.add_constraint(tab,srd)
    tab.clear()

t.solve()""""""
"""

# given solution: z=11, [0,3,1] - correct