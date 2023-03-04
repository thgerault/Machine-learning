import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor, BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
import math
# Pour normaliser et standardiser les données
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns


class Gradient_boosting():
    def __init__(self, X, Y, n_apprenant_faible, learning_rate, n_boostrap):
        self.X=X
        self.Y=Y
        self.n_apprenant_faible=n_apprenant_faible
        self.learning_rate=learning_rate
        self.n_boostrap=n_boostrap

    def choix(self):
        """
        Cette fonction sert à retourner une liste de tout les apprenants faibles que l'on à souhaités en renseignant l'argument : n_apprenant_faible. 
        Le but est d'avoir la possibilité de mettre autant d'apprenant faible que l'ont souhaite parmi les 4 possibles. 
        Par exemple, si je veux seulement 3 RandomForest, en ayant renseigner la liste [3,0,0,0], alors cette fonction retournera : 
        [RandomForestRegressor(),RandomForestRegressor(),RandomForestRegressor()]
        """
        # Voici les 4 types d'apprenants faible possible pour ce programme du Gradient boosting
        apprenant_faible=[RandomForestRegressor(),BaggingRegressor(),DecisionTreeRegressor(),KNeighborsRegressor()] 
        apprenant=[]
        for k in range (len(self.n_apprenant_faible)):
            nb=0
            while nb != self.n_apprenant_faible[k]:
                apprenant.append(apprenant_faible[k])
                nb+=1
        
        return apprenant
    
    def regle_de_decision(self, y_pred):
        """
        Cette fonction va servir à appliquer une règle qui nous permettra de convertir nos valeurs calculées entier. 
        0,1,2 pour trois classes par exemple.
        """
        K=list(set(self.Y))
        y_copy=np.copy(y_pred)
        for k in range(len(K)):
            for i in range(len(y_pred)):
                if y_copy[i]<=K[0]:
                    y_copy[i]=K[0]
                elif y_copy[i]>K[k-1] and y_copy[i]<=K[k]:
                    y_copy[i]=K[k]
        
        # Mise sous forme d'entier
        y_copy=[int(y_copy[i]) for i in range(len(y_pred))] # Convertion en entier car la bouble renvoie des float : 0.0, 1.0, ...
        return y_copy

    def entrainement_gradient_boosting(self,train_test):

        Xtrain, Xtest, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=self.n_boostrap)
        
        scaler=StandardScaler()
        X_train=scaler.fit_transform(Xtrain)
        X_test=scaler.fit_transform(Xtest)

        """
        Choix : Soit on choisit d'entrainer sur les données d'entrainement soit sur les données tests.
        Ceci est mis en place pour nous permettre de calculer l'entropie croisée sur les données tests
        """
        if train_test=="train":
            X=np.copy(X_train)
            Y=np.copy(Y_train)
        elif train_test=="test":
            X=np.copy(X_test)
            Y=np.copy(Y_test)            

        """
        Initialisation de l'algorithme
        """
        K=list(set(Y))
        F=[]    
        F.append([0 for i in range(len(Y))]) # initialisation, tt le monde à 0
        P=[1/len(K) for i in range(len(Y))] # pkm initial = 1/K
        
        # Programmation de l'algorithme de Gradient Boosting
        liste_apprenant=[]
        apprenant=Gradient_boosting.choix(self) # Récupération des apprenants faibles à utiliser
        entropie_croiser=[]

        for m in range(len(apprenant)):

            """
            Etape 1 : Grace à la fonction softmax, on calculer la probabilité conditionnelle pk(xi) d'appartenance à la classe k.
            """
            denominateur=0
            if m==0:
                new_K=[0 for i in range(len(K))] # necessite de différencier le cas m=0 car au départ on a que des 0
            else: new_K=list(set(F[m]))

            for k in range(len(K)):
                denominateur+= np.exp(new_K[k]) # Somme(exp(Fkm))

            # Calcul de pk(xi), la fonction softmax
            for j in range(len(Y)):
                P[j]=round(np.exp(F[m][j])/denominateur,5) # exp(Fkm)/Somme(exp(Flm)) pour l = 1,...,K

            """
            Etape 2 : Calcul des résidus, descente de gradient. Il s'agit tout simplement de la différence entre 
            les Yi et la probabilité associé d'appartenir à la classe Yi
            """
            residu=[Y[i]-P[i] for i in range(len(Y))]
            
            """
            Etape 3 : Entrainement des apprenants faibles sur les résidus. On stock en même temps les apprenants qui sont entrainer. 
            """
            clf=apprenant[m]
            clf.fit(X,residu)
            pred=clf.predict(X)
            liste_apprenant.append(clf)
            
            """
            Etape 4 : Mise à jour des nouvelles valeur et introduction du learning rate. A noter que la fonction
            round est là juste pour arrondir à 5 chiffre pour éviter d'avoir des valeurs > K ce qui planterait tout l'algo
            """
            F.append([F[m][i] + self.learning_rate*round(pred[i],5) for i in range(len(Y))])
            
            """
            Etape 5 : Calcul de l'entropie croisée pour la fonction de perte
            """
            classe=Gradient_boosting.regle_de_decision(self, F[-1])
            loss=[]
            for i in range(len(Y)):
                if Y[i] != classe[i]:
                    loss.append(-math.log(P[i]))

            entropie_croiser.append(sum(loss))
            
        return m, F, entropie_croiser, liste_apprenant, X_train, X_test, Y_train, Y_test

    def predictions(self):
        """
        Cette fonction va effectuer des prédictions sur les données tests. Pour cela, on récupère tout les apprenants entrainer dans la fonction
        précédentes. Puis on les entrainent sur les données tests en oubliant pas le learning rate. On effectue une somme de ces prédictions et
        on applique à cette somme notre règle de décision.
        """
        m, F, entropie_croiser, liste_apprenant, X_train, X_test, Y_train, Y_test = Gradient_boosting.entrainement_gradient_boosting(self,"train")
        y_pred_estimate=np.zeros(len(Y_test))

        entropie_croiser_test=[]
        for learning in liste_apprenant:
            pred=learning.predict(X_test)
            for i in range(len(y_pred_estimate)):
                y_pred_estimate[i]+=self.learning_rate*pred[i]

        y_pred_estimate=Gradient_boosting.regle_de_decision(self, y_pred_estimate)
    
        return m, X_train, X_test, Y_train, Y_test, y_pred_estimate

    
    def matrice_confusion(self):
        """
        Comme son nom l'indique, on va plot une matrice de confusion et également celle du Gradient boosting de
        sklearn avec les même paramètre pour pouvoir comparer notre algorithme.
        """

        m, X_train, X_test, Y_train, Y_test, y_pred_estimate = Gradient_boosting.predictions(self)

        # On calcul l'accuracy
        print("Voici l'accuracy de cette algorithme : ", accuracy_score(Y_test,y_pred_estimate))

        # Entrainement et prédictions avec le GB de sklearn
        verification=GradientBoostingClassifier(n_estimators=m, learning_rate= self.learning_rate)
        verification.fit(X_train, Y_train)
        y_pred=verification.predict(X_test)

        fig,axes = plt.subplots(1,2,figsize=(15,5))
        conf_matrix_estimate=confusion_matrix(Y_test,y_pred_estimate)
        sns.heatmap(conf_matrix_estimate, annot = True ,cmap="Blues",ax=axes[0])
        axes[0].set_xlabel("Predicted label")
        axes[0].set_ylabel("True label")
        axes[0].set_title("Gradient Boosting estimé")

        conf_matrix=confusion_matrix(Y_test,y_pred)
        sns.heatmap(conf_matrix, annot = True ,cmap="Reds",ax=axes[1])
        axes[1].set_xlabel("Predicted label")
        axes[1].set_ylabel("True label")
        axes[1].set_title("Gradient Boosting de sklearn")

    
    def loss_function(self):
        """
        On calcul ici la fonction de perte associé à cette algo. Pour cela on récupère le calcul
        de l'entropie croisée calculer dans la fonction d'entrainement du modele. Ensuite, on applique 
        également la fonction d'entrainement sur les données tests pour avoir également les erreurs tests sur le même graphique
        """

        _,_, entropie_croiser,_, _,_,_,_=Gradient_boosting.entrainement_gradient_boosting(self,"train")
        _,_, entropie_croiser_test,_,_,_,_,_ = Gradient_boosting.entrainement_gradient_boosting(self,"test")
        
        fct_perte=pd.DataFrame()
        fct_perte["Loss train"]=entropie_croiser
        fct_perte["Validation test"]=entropie_croiser_test
        fct_perte.plot()
        plt.title("Fonction de perte estimé par entropie croisée")
        

if __name__ == '__main__':
    start=time.time()

    iris=datasets.load_iris() # Chargement du jeu de données du package sklearn
    X_iris=iris.data
    y_iris=iris.target

    Gradient_boosting(X_iris, y_iris, [0,0,0,6],0.2,0.2).matrice_confusion()
    Gradient_boosting(X_iris, y_iris, [0,0,0,6],0.2,0.2).loss_function()


    end=time.time()
    print("Exécution du programme : ",end-start,"secondes")