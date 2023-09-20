#on va generer une dataset x,y
#on va generer une dataset x,y
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X,y =make_blobs(n_samples=100, n_features=2,centers=2,random_state=0)
y=y.reshape((y.shape[0],1))
print("demension de X:",X.shape)
print("dimensions de y:",y.shape)
plt.scatter(X[:,0],X[:,1],c=y,cmap='summer')
plt.show()

#fonction d'initialisation
def initialisation(X):
    #la dimension  de X.shape[1]
    W=np.random.randn(X.shape[1],1)
    #on passe un nombre reel
    b=np.random.randn(1)
    return (W,b)

#fonction de notre modele
def model(X,W,b):
     Z=X.dot(W)+b
     A=1/(1+np.exp(-Z))
     return A
def prediction(X,w,b):
    A = model(X,w,b)
    print(A)# la probabilté
    return A>=0.5
#fonction d'erreur
def Error(A,y):
    return 1/2*(y-A)**2

#fonction gradient
def gradients(A,X,y):
    p1=np.dot((y-A).T,(A-1))
    p2=np.dot(p1,A.T);
    dw=np.dot(X.T,p2.T)
    db=-(A-y)*(A-1)*A
    return dw,db
W,b=initialisation(X)
A=model(X,W,b)
print("A")
print(A.shape)
print("XT:",X.T.shape)
dw,db=gradients(A,X,y)
print("w",W.shape)
print("wd",dw.shape)
def update(dw,db,W,b,learning_rate):
    W=W-learning_rate*dw
    b=b-learning_rate*db
    return W,b
def artificial_neuron(X,y,learning_rate=0.1, n_iter=100):
    W,b=initialisation(X)
    Erreur=[]
    for i in range(n_iter):
        A=model(X,W,b)
        Erreur.append(Error(A,y))
        dw,db=gradients(A,X,y)
        W,b=update(dw,db,W,b, learning_rate)
    plt.plot(Erreur)
    plt.show()
artificial_neuron(X,y)

w_p,b_p=artificial_neuron(X,y,0.1,100)
print(w_p,b_p)