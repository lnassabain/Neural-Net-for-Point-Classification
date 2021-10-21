import pandas as pd 
from sklearn.model_selection import train_test_split
import numpy as np
from utility import Utility as uti
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns

gaussian_df = pd.read_csv("./gaussian_data.csv")
test_final_df = gaussian_df.sample(frac = 0.2, random_state=42)
gaussian_df = gaussian_df.drop(test_final_df.index)


X = gaussian_df.iloc[:, :2]
y = gaussian_df.iloc[:, 2:]

X_test_final = test_final_df.iloc[:, :2]
y_test_final = test_final_df.iloc[:, 2:]

y = pd.get_dummies(y)
y_test_final = pd.get_dummies(y_test_final)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

class NeuralNet:
    
    def __init__(self, X_train = None, y_train = None, X_test = None, y_test = None,hidden_layer_sizes = (4,)\
    , activation = uti.identity, learning_rate=0.01, epoch=200):
        self.X_train = X_train.to_numpy()
        self.y_train = y_train.to_numpy()
        self.X_test = X_test.to_numpy()
        self.y_test = y_test.to_numpy()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.nb_input = np.shape(self.X_train)[1] #nbr d'unités en entrée
        self.nb_output = np.shape(self.y_train)[1] #nbr d'unités en sortie
        self.n_layers = len(hidden_layer_sizes)
        self.W = [None] * (self.n_layers + 1) #liste de matrices de poids
        self.B = [None] * (self.n_layers + 1) #liste de matrices de biais
        self.Z = [None] * (self.n_layers + 1) #liste de mat. d'entrees pond.
        self.A = [None] * (self.n_layers + 1) #lsite de mat. d'acivation 
        self.learning_rate = learning_rate
        self.activation = activation
        self.nb_epoch = epoch
        self.__weights_initialization(X_train, y_train)
        self.df = [None] * (self.n_layers + 1)

 #-------------------------------------------------------------------# 
    #Fonction qui initialise les tableaux W et B (parametres du modèle)

    def __weights_initialization(self, X, y):
        entree = self.nb_input #nbr d'unités dans la couche d'entree
        sortie = self.nb_output #nbr d'unités dans la couche de sortie
        
        m1 = np.random.uniform(low=-1, high=1, size=(self.hidden_layer_sizes[0], entree))
        m2 = np.random.uniform(low=-1, high=1, size=(sortie, self.hidden_layer_sizes[self.n_layers-1]))
        self.W[0] = m1
        self.W[self.n_layers] = m2

        for i in range(1, self.n_layers):
            m = np.random.uniform(low=-1, high=1, size=(self.hidden_layer_sizes[i], self.hidden_layer_sizes[i-1]))
            self.W[i] = m

        for i in range(self.n_layers):
            m = np.random.uniform(low=-1, high=1, size=(self.hidden_layer_sizes[i], 1))
            self.B[i] = m
        
        self.B[self.n_layers] = np.random.uniform(low=-1, high=1, size=(sortie,1))


 #------------------------------------------------------------------#
    #Fonction de passe-avant
    #Si l'on veut utiliser feed_forward en dehors de la fonction fit()
    #il faut ajouter le code suivant au début de la fonction:
    #X = X.reshape(-1,1)
    #y = y.reshape(-1,1)

    def __feed_forward(self, X, y):
        self.Z[0] = np.dot(self.W[0], X) + self.B[0]
        self.A[0],self.df[0] = self.activation(self.Z[0])

        L = self.n_layers
        for l in range(1, L):
            self.Z[l] = np.dot(self.W[l], self.A[l-1]) + self.B[l]
            self.A[l],self.df[l] = self.activation(self.Z[l])
        
        self.Z[L] = np.dot(self.W[L], self.A[L-1]) + self.B[L]
        self.A[L] = uti.softmax(self.Z[L])

        erreur = uti.cross_entropy_cost(self.A[L], y)
        return erreur,self.A[L]

        

 #------------------------------------------------------------------#
    #Fonction de retro-propagation
    #Si l'on veut utiliser backward_pass en dehors de la fonction fit()
    #il faut ajouter le code suivant au debut de la fonction:
    #y = y.reshape(-1,1)
    #X = X.reshape(-1,1)

    def __backward_pass(self, X, y):
        delta = [None] * (self.n_layers + 1)
        dW = [None] * (self.n_layers + 1)
        db = [None] * (self.n_layers + 1)
        
        eta = self.learning_rate
        L = self.n_layers
        
        delta[L] = np.subtract(self.A[L], y)
        dW[L] = np.dot(delta[L], self.A[L-1].transpose())
        db[L] = delta[L]
        for l in reversed(range(1,L)):
            delta[l] = np.multiply(np.dot(self.W[l+1].transpose(), delta[l+1]), self.df[l])
            dW[l] = np.dot(delta[l], self.A[l-1].transpose())
            db[l] = delta[l]

        delta[0] = np.multiply(np.dot(self.W[1].transpose(), delta[1]), self.df[0])
        dW[0] = np.dot(delta[0], X.transpose())
        db[0] = delta[0]

        for i in range(L+1):
            self.W[i] = np.subtract(self.W[i], eta*dW[i])
            self.B[i] = np.subtract(self.B[i], eta*db[i])
        
 #------------------------------------------------------------------#
    #Fonction renvoie la prédiction du modèle pour une instance x.
    #On initialise un vecteur de taille (nb_unites_dans_couche_de_sortie,1)   
    #à 0 pour qu'on puisse utiliser la fonction feed_forward

    def predict(self, x):
        y = np.zeros((self.nb_output,1))
        x = x.reshape(-1, 1)
        pred = self.__feed_forward(x, y)[1]
        return pred

 #------------------------------------------------------------------#
    #Fonction execute 1 epoque d'entrainement
    #Renvoie la moyenne des erreurs sur le jeu de données d'entrainement
    #et la moyenne des erreurs sur le jeu de données de test

    def fit(self):
        self.X_train, self.y_train = shuffle(self.X_train, self.y_train)
        
        nbr_inst1 = np.shape(self.X_train)[0]
        nbr_inst2 = np.shape(self.X_test)[0]
        err1 = [None] * nbr_inst1
        for i in range(nbr_inst1):
            X = self.X_train[i].reshape(-1,1)
            y = self.y_train[i].reshape(-1,1)  
            err1[i] = self.__feed_forward(X,y)[0]
            self.__backward_pass(X,y) 
        
        erreur_train = sum(err1)/nbr_inst1

        err2 = [None] * nbr_inst2
        for i in range(nbr_inst2):
            X = self.X_test[i].reshape(-1,1)
            y = self.y_test[i].reshape(-1,1)
            err2[i] = self.__feed_forward(X,y)[0]
        
        erreur_test = sum(err2)/nbr_inst2

        return erreur_train,erreur_test
#------------------------------------------------------------------#
#--------------------------end of class----------------------------#

#Fonction qui entraine le reseau pendant ep epoques et qui renvoie 2 tableaux 
#contenant l'erreur sur le jeu d'entrainement et sur le jeu de test 
#pour chaque epoque

def train(neural_net, ep):
    train_error = [None] * ep
    test_error = [None] * ep
    for i in range(ep):
        train_error[i],test_error[i] = neural_net.fit()

    return train_error, test_error


#-------------------------------------------------------------------#
#Fonction calculant les predictions d'un modèle "neural_net" pour chaque 
#instance de X_test_final et qui convertit les etiquettes 
#y_test_final en tableau Numpy
#La fonction renvoie les predictions arrondies a 2 decimales et 
#les etiquettes ainsi transformées

def final_pred(neural_net, X_test_final, y_test_final):
    X_test_final = X_test_final.to_numpy()
    y_test_final = y_test_final.to_numpy()

    nb_inst = np.shape(X_test_final)[0]
    y_pred = np.zeros((nb_inst, 3))
    for i in range(nb_inst):
        y_pred[i] = neural_net.predict(X_test_final[i]).transpose()

    return np.round(y_pred, decimals=2), y_test_final


#-------------------------------------------------------------------#
#Fonction qui affiche l'evolution de l'erreur en fonction des epoques.
#Faite pour faciliter l'affichage de l'evolution de l'erreur.
#On doit creer un array contenant les entiers de 0 à nbr d'époques
#qui nous servira de valeurs sur l'axe des abscisses

def evolution_plot(train_error, test_error):
    assert(len(train_error) == len(test_error)), "evolution_plot:\
    la taille du tableau contenant l'erreur sur le jeu d'entrainement pour chaque epoque \
    doit être egale à la taille du tableau contenant l'erreur sur le jeu de test"
    length = len(train_error)
    x_tab = [None] * length
    for i in range(length):
        x_tab[i] = i+1
        
    plt.plot(x_tab, train_error, label = 'Train') 
    plt.plot(x_tab, test_error, label = 'Test')
    plt.legend()
    plt.title('Evolution of error during training')
    plt.xlabel('Epoch of training')
    plt.ylabel('Error')
    plt.show()

#-------------------------------------------------------------------#
#Fonction renvoyant l'indice de l'élément le plus grand de l'array 'row'

def max_row(row):
    return np.where(row==np.amax(row))[0][0]

#------------------------------------------------------------------#
#Fonction qui renvoie le ratio de prédictions correctes sur un jeu de données
#par rapport au nombre total de prédictions

def ratio(y_pred, y_actual):
    assert (np.shape(y_pred) == np.shape(y_actual)), "ratio:\
    y_pred et y_actual devraient avoir la même taille"
    total = np.shape(y_pred)[0]
    correct = 0
    for i in range(total):
        if (max_row(y_pred[i]) == max_row(y_actual[i])):
            correct += 1
        
    rat = correct/total
    return rat

#------------------------------------------------------------------#
#Fonction prenant en parametres une matrice contenant les predictions
#et une matrice contenant les valeurs cibles.
#La fonction renvoie la matrice de confusion

def mat_conf(y_pred, y_actual):
    assert(np.shape(y_pred) == np.shape(y_actual)), "mat_conf:\
    y_pred et y_actual devraient avoir la meme taille"
    nb_class = np.shape(y_pred)[1]
    mat = np.zeros((nb_class, nb_class), dtype=int)
    nb_inst = np.shape(y_pred)[0]
    for i in range(nb_inst):
        p = max_row(y_pred[i])
        a = max_row(y_actual[i])
        mat[p][a] += 1
    
    return mat

#===================================================================#
#===========================PARTIE TEST=============================#
#===================================================================#

#reseau = NeuralNet(X_train, y_train, X_test, y_test, hidden_layer_sizes=(4,3,2)\
#,activation=uti.tanh, learning_rate=0.01, epoch=200)

#train_error, test_error = train(reseau, reseau.nb_epoch)
#evolution_plot(train_error, test_error)

#y_pred, y_actual = final_pred(reseau,X_test_final, y_test_final)
#r = ratio(y_pred, y_actual)
#print(r)

#mat_con = mat_conf(y_pred, y_actual)
#class_names = ['class-0', 'class-1', 'class-2']
#plt.figure(figsize = (8,8))
#sns.set(font_scale=2) 
#ax = sns.heatmap(mat_con, annot=True, annot_kws={"size": 30}, 
#cbar=False, cmap='Blues', fmt='d', 
#xticklabels=class_names, yticklabels=class_names)
#ax.set(title='', xlabel='Actual', ylabel='Predicted')
#plt.show()

#===================================================================#
#=======================FIN DE LA PARTIE TEST=======================#
#===================================================================#