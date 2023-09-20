import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Chargement des données CIFAR-10
X, y = fetch_openml('CIFAR_10', version=1,parser='auto', return_X_y=True)
# Normalisation des données
X = X / 255.0
# Division des données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Hyperparamètres du réseau de neurones
input_size = 32*32*3  # Taille de l'entrée (nombre de pixels dans chaque image)
hidden_size = 128  # Nombre de neurones dans la couche cachée
output_size = 10  # Nombre de classes (chiffres de 0 à 9)
learning_rate = 0.001  # Taux d'apprentissage
momentum = 0.9  # Taux de momentum
num_epochs = 10  # Nombre d'époques d'entraî


# Initialisation des poids et des biais du réseau de neurones
W1 = np.random.randn(input_size, hidden_size) * 0.01  # Poids de la couche cachée
b1 = np.zeros((1, hidden_size))  # Biais de la couche cachée
W2 = np.random.randn(hidden_size, output_size) * 0.01  # Poids de la couche de sortie
b2 = np.zeros((1, output_size))  # Biais de la couche de sortie




def forward(X, W1, b1, W2, b2):
    """
    Propagation avant dans le réseau de neurones.

    Parameters
    ----------
    X : numpy array
        Données d'entrée.
    W1 : numpy array
        Poids de la couche cachée.
    b1 : numpy array
        Biais de la couche cachée.
    W2 : numpy array
        Poids de la couche de sortie.
    b2 : numpy array
        Biais de la couche de sortie."""
            # Propagation avant dans la couche cachée
    Z1 = np.dot(X, W1) + b1
    A1 = 1 / (1 + np.exp(-Z1))  # Fonction d'activation sigmoïde

    # Propagation avant dans la couche de sortie
    Z2 = np.dot(A1, W2) + b2
    A2 = 1 / (1 + np.exp(-Z2))  # Fonction d'activation sigmoïde

    return A2, (Z1, A1, W1, b1, W2, b2)


def backward(X, y, A2, cache, learning_rate=0.001, momentum=0.9):
    """
    Rétropropagation dans le réseau de neurones.

    Parameters
    ----------
    X : numpy array
        Données d'entrée.
    y : numpy array
        Valeurs réelles.
    A2 : numpy array
        Prédictions du réseau de neurones.
    cache : tuple
        Valeurs intermédiaires de la propagation avant.
    learning_rate : float, optional
        Taux d'apprentissage (par défaut 0.001).
    momentum : float, optional
        Taux de momentum (par défaut 0.9).

    Returns
    -------
    W1_grad : numpy array
        Gradient des poids de la couche cachée.
    b1_grad : numpy array
        Gradient des biais de la couche cachée.
    W2_grad : numpy array
        Gradient des poids de la couche de sortie.
    b2_grad : numpy array
        Gradient des biais de la couche de sortie.
    """
    m = X.shape[0]

    # Récupération des valeurs intermédiaires de la propagation avant
    Z1, A1, W1, b1, W2, b2 = cache

    # Calcul des gradients de la couche de sortie
    dZ2 = A2 - y
    dW2 = 1 / m * np.dot(A1.T, dZ2)
    db2 = 1 / m * np.sum(dZ2, axis=0, keepdims=True)

    # Calcul des gradients de la couche cachée
    dZ1 = np.dot(dZ2, W2.T) * (1 - np.power(A1, 2))
    dW1 = 1 / m * np.dot(X.T, dZ1)
    db1 = 1 / m * np.sum(dZ1, axis=0, keepdims=True)
    return  dW2 , dW1,db2,db1

def cost(y_pred, y):
    """
    Fonction de coût logistique.

    Parameters
    ----------
    y_pred : numpy array
        Prédictions du réseau de neurones.
    y : numpy array
        Valeurs réelles.

    Returns
    -------
    cost : float
        Coût logistique.
    """

    m = y.shape[0]
    cost = -1 / m * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return cost



def train(X, y, W1, b1, W2, b2, learning_rate, momentum, num_epochs):
    """
    Entraînement du réseau de neurones.Parameters
    ----------
    X : numpy array
        Données d'entrée.
    y : numpy array
        Valeurs réelles.
    W1 : numpy array
        Poids de la couche cachée.
    b1 : numpy array
        Biais de la couche cachée.
    W2 : numpy array
        Poids de la couche de sortie.
    b2 : numpy array
        Biais de la couche de sortie.
    learning_rate : float, optional
        Taux d'apprentissage (par défaut 0.001).
    momentum : float, optional
        Taux de momentum (par défaut 0.9).
    num_epochs : int, optional
        Nombre d'époques d'entraînement (par défaut 10).

    Returns
    -------
    W1 : numpy array
        Poids de la couche cachée optimisés.
    b1 : numpy array
        Biais de la couche cachée optimisés.
    W2 : numpy array
        Poids de la couche de sortie optimisés.
    b2 : numpy array
        Biais de la couche de sortie optimisés.
    costs : list
        Liste des coûts logistiques à chaque époque d'entraînement.
    """
    costs = []  # Liste des coûts logistiques à chaque époque d'entraînement

    # Initialisation des vélocités pour la descente de gradient avec momentum
    vW1 = np.zeros(W1.shape)
    vb1 = np.zeros(b1.shape)
    vW2 = np.zeros(W2.shape)
    vb2 = np.zeros(b2.shape)

    for epoch in range(num_epochs):
        # Propagation avant
        A2, cache = forward(X, W1, b1, W2, b2)

        # Calcul du coût logistique
        cost=cost(A2, y)
        costs.append(cost)

        # Rétropropagation
        W1_grad, b1_grad, W2_grad, b2_grad = backward(X, y, A2, cache, learning_rate, momentum)
        vW1 = momentum * vW1 - learning_rate * W1_grad
        vb1 = momentum * vb1 - learning_rate * b1_grad
        vW2 = momentum * vW2 - learning_rate * W2_grad
        vb2 = momentum * vb2 - learning_rate * b2_grad

        W1 += vW1
        b1 += vb1
        W2 += vW2
        b2 += vb2

    return W1, b1, W2, b2,costs
def cost(y_pred, y):
    """
    Fonction de coût logistique.

    Parameters
    ----------
    y_pred : numpy array
        Prédictions du réseau de neurones.
    y : numpy array
        Valeurs réelles.

    Returns
    -------
    cost : float
        Coût logistique.
    """
    m = y.shape[0]
    cost = -1 / m * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return cost


def test(X, y, W1, b1, W2, b2):
    """
    Test du réseau de neurones sur l'ensemble de test.

    Parameters
    ----------
    X : numpy array
        Données d'entrée.
    y : numpy array
        Valeurs réelles.
    W1 : numpy array
        Poids de la couche cachée.
    b1 : numpy array
        Biais de la couche cachée.
    W2 : numpy array
        Poids de la couche de sortie.
    b2 : numpy array
        Biais de la couche de sortie.

    Returns
    -------
    accuracy : float
        Accuracy des prédictions.
    """
    # Propagation avant

    A2, _ = forward(X, W1, b1, W2, b2)

    # Calcul de l'accuracy des prédictions
    accuracy = np.mean(np.argmax(A2, axis=1) == np.argmax(y, axis=1))

    return accuracy



# Entraînement du réseau de neurones
W1, b1, W2, b2, costs = train(X_train, y_train, W1, b1, W2, b2, learning_rate, momentum, num_epochs)

# Test du réseau de neurones sur l'ensemble de test

accuracy = test(X_test, y_test, W1, b1, W2, b2)

# Affichage de l'accuracy des prédictions
print(f'Accuracy : {accuracy:.2f}')

"""# Affichage de la courbe d'apprentissage
plt.plot(costs)
plt.ylabel('Coût logistique')
plt.xlabel('Époque dentraînement')
plt.show()
"""
