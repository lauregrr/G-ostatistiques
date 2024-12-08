# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:01:01 2024

@author: leoni
"""

import numpy as np
import matplotlib.pyplot as plt

import lib_gs as gs
import math as m
import pandas as pd

#Etape 1: calculer la nuée du variogramme
def nuee_vario(x_obs, y_obs, z_obs):
    
    """
    Fonction pour calculer la nuée variographique de nos observations
    
    Parameters
    ----------
    x_obs, y_obs, z_obs : vecteurs colonnes qui représentent les coordonnées des observations sur 
    lesquelles vont être effectuées l'interpolation

    Returns
    -------
    Hij : array qui stocke les distances entre chaque couple de points
    DZ : 

    """
    # Créations de listes vides pour stocker nos valeurs
    Hij = []
    Dz = [] 
    
    for i in range (len(x_obs)): 
        for j in range (len(x_obs)): 
            
            # Test: pour ne pas recalculer les valeurs entre les couples 2 fois 
            # On calcul qu'une seule fois la distance et le variogramme entre les 
            # points i/j et j/i
            if i<=j: 
                h = np.sqrt((x_obs[i]-x_obs[j])**2+(y_obs[i]-y_obs[j])**2)
                d = (1/2)*(z_obs[i]-z_obs[j])**2
                
                Hij.append(h)
                Dz.append(d)
                
            else: 
                pass
    
    # Plot de la nuée variographique
    plt.plot(Hij,Dz,'.')
    plt.xlabel(r'$h_{i,j}$')
    plt.ylabel(r'$\Delta z_{i,j}^2/2$')
    plt.grid()

    return Hij, Dz        

#Etape 2: calcul du variogramme expérimentale: 
def vario_exp(Hij, Dz, d):
    """
    Fonction pour calculer variogramme expérimentale à partir de notre nuée variographique
    Parameters
    ----------
    Hij, Dz : listes qui stockent la distance et le variogramme entre chaque couple de points
    d : integer qui définit le nombre de points par paquet pour calculer notre variogramme expérimentale

    Returns
    -------
    H_exp, Gam_exp : listes qui stockent les distances et variogrammes moyens moyens pour chaque paquet de points

    """
    # Création d'un dataFrame pour rendre plus facile le tri des points en fonction de leur valeur
    df = pd.DataFrame({'dist': Hij, 'Dz': Dz})
    df_trie = df.sort_values(by='dist')
    
    H_exp = []
    Gam_exp = []
        
    for i in range (0, len(Dz), d):
        # Sélectionner les d premières valeurs de la colonne spécifique
        dist500 = df_trie['dist'].iloc[i:(d+i)]
        Dz500 = df_trie['Dz'].iloc[i:(d+i)]
    
        # Calculer des moyenns pour chaque paquets de d points
        moyDist = dist500.mean()
        moyDz = Dz500.mean()
        
        H_exp.append(moyDist[0])
        Gam_exp.append(moyDz[0])
        
    plt.plot(Hij,Dz,'.',label='Nuéee')
    plt.plot(H_exp,Gam_exp,'r',label='Var. exp.')
    plt.xlabel(r'$h$')
    plt.ylabel(r'$\Delta z_i^2/2$')
    plt.legend()
    plt.grid()
    
    return H_exp, Gam_exp

#Etape 3: moindre carré - modèle cubique

#Détermination de B
def gamma(H_exp, a, C):
    Gam = []
    for h_exp in H_exp: 
        if h_exp>a: 
            gam = C
            Gam.append(gam)
            
        else: 
            gam = C*(7*(h_exp**2/a**2)-(35/4)*(h_exp**3/a**3)+(7/2)*(h_exp**5/a**5)-(3/4)*(h_exp**7/a**7))
            Gam.append(gam)
        
    
    Gamm = np.array(Gam)
    return Gamm

#Détermination de A
def gamma_da(H_exp, a, C):
    """
    Dérivée de la fonction gamma en fonction de a
    """
    Gam = []
    for h_exp in H_exp: 
        if h_exp>a: 
            gam = 0
            Gam.append(gam)
            
        else: 
            gam = C*(-14*(h_exp**2/a**3)+(105/4)*(h_exp**3/a**4)-(35/2)*(h_exp**5/a**6)+(21/4)*(h_exp**7/a**8))
            Gam.append(gam)
        
    
    Gamm = np.array(Gam)
    return Gamm

def gamma_dC(H_exp, a, C):
    """
    Dérivée de la fonction gamma en fonction de C
    """
    Gam = []
    for h_exp in H_exp: 
        if h_exp>a: 
            gam = 1
            Gam.append(gam)
            
        else: 
            gam = (7*(h_exp**2/a**2)-(35/4)*(h_exp**3/a**3)+(7/2)*(h_exp**5/a**5)-(3/4)*(h_exp**7/a**7))
            Gam.append(gam)
        
    
    Gamm = np.array(Gam)
    return Gamm
 
def vario_cubique (Hij, Dz, H_exp, Gam_exp):
    """
    Fonction qui calcul notre variogramme cubique à partir des données expérimentales

    Parameters
    ----------
    Hij, Dz : listes qui stockent la distance et le variogramme entre chaque couple de points
    
    H_exp, Gam_exp : listes qui stockent les distances et variogrammes moyens moyens pour chaque paquet de points
    
    Returns
    -------
    Gam_cub : variogramme cubique, ensemble de points sur la forme d'un array qui représente 
    la droite qui apparoche notre variogramme expérimentale
    a0 : float 
    C0 : float

    """
    #Initialisation des paramètres 
    #Déterminantion de a0 et C0 = donne les valeurs du premier X: 
    
    indice_max = np.argmax(Gam_exp)
    C0 = Gam_exp[indice_max]
    a0 = H_exp[indice_max]
    
    #faire tourner les moindres carrés 
    for i in range(10): 
    
        A = np.column_stack((gamma_da(H_exp, a0, C0), gamma_dC(H_exp, a0, C0))) 
        B = Gam_exp - gamma(H_exp, a0, C0)
        B = B.reshape((-1,1))
        # print(B.shape)
        # print(A.shape)
        
        X_chap = np.linalg.inv(np.transpose(A)@A)@np.transpose(A)@B
        a0 = a0 + X_chap[0,0]
        C0 = C0 + X_chap[1,0]
        
        
    Gam_cub = gamma(H_exp, a0, C0)
    plt.plot(H_exp,Gam_cub,'g',label=r'$\gamma_{cub}$')
    plt.xlabel(r'$h$')
    plt.ylabel(r'$\Delta z_i^2/2$')
    plt.legend()
    plt.grid()
    
    return Gam_cub, a0, C0
    
    #Etape 4 - modèle linéaire
    
def gamma_lin(H_exp, Gam_exp):
    """
    Fonction qui calcul notre variogramme linéaire à partir des données expérimentales

    Parameters
    ----------
    H_exp, Gam_exp : listes qui stockent les distances et variogrammes moyens moyens pour chaque paquet de points
    
    Returns
    -------
    Gam : variogramme linéaire, ensemble de points sur la forme d'un array qui représente 
    la droite qui apparoche notre variogramme expérimentale

    """
    
    H_exp = np.array((H_exp))
    H_exp = H_exp.reshape((-1,1))
    Gam_exp = np.array((Gam_exp))
    Gam_exp = Gam_exp.reshape((-1,1))
    
    w = np.linalg.inv(np.transpose(H_exp) @ H_exp) @ np.transpose(H_exp) @ Gam_exp
    w = w[0][0]
    
    Gam = []
    for h_exp in H_exp: 
        gam = w*h_exp
        Gam.append(gam)
        
    Gamm = np.array(Gam)
    
    plt.plot(H_exp,Gamm,'b',label=r'$\gamma_{lin}$')
    plt.xlabel(r'$h$')
    plt.ylabel(r'$\Delta z_i^2/2$')
    plt.legend()
    plt.grid()
    
    return Gamm
    
#gamma_lin = gamma_lin(H_exp, w)
    

#Etape 5 - Krigeage

#Calcul de la matrice des variogrammes "fixes"

def vario_ajustee(x1, y1, x2, y2, a0, C0):
    """
    Fonction qui calcule les paramètres du krigeage, calcul du variogramme entre deux points
    en fonction des paramètres a0 et C0 déterminé par moindre carré
    Parameters
    ----------
    x1, y1 : float, coordonnées du point i 
    x2, y2 : float, coordonnées du point j 
    
    a0, C0 : float

    Returns
    -------
    gam : float

    """
    
    d = m.sqrt((x1-x2)**2+(y1-y2)**2)
    gam = C0*(7*(d**2/a0**2)-(35/4)*(d**3/a0**3)+(7/2)*(d**5/a0**5)-(3/4)*(d**7/a0**7))
    
    return gam
    

def krigeage (x_obs, y_obs, z_obs, a0, C0): 
    """
    Calcul de la matrice A du modèle du krigeage
    
    x_obs, y_obs, z_obs : vecteurs colonnes qui représentent les coordonnées des observations sur 
    lesquelles vont être effectuées l'interpolation
    
    RETURN: 
        A : array
    """
    
    A = np.ones((len(x_obs)+1, len(x_obs)+1))
    A[len(x_obs), len(x_obs)] = 0
    
    for i in range (len(x_obs)): 
        for j in range (len(x_obs)): 
            
            if i==j: 
                A[i,j]=0
                
            else: 
                if A[i,j]==1: 
                    A[i,j] = vario_ajustee(x_obs[i], y_obs[i], x_obs[j], y_obs[j] , a0, C0)
                    A[j,i] = vario_ajustee(x_obs[i], y_obs[i], x_obs[j], y_obs[j] , a0, C0)
                
    return A

#test = krigeage (x_obs, y_obs, z_obs, a0, C0)

def krig_grille(x, y, x_obs, y_obs, a0, C0): 
    """
    Calcul de la matrice B du modèle du krigeage: calcul le variogramme ajusté pour chaque couple de point 
    et le stocke dans une matrice
    
    Parameters
    ----------
    x_obs, y_obs, z_obs : vecteurs colonnes qui représentent les coordonnées des observations sur 
    lesquelles vont être effectuées l'interpolation
    
    a0, C0: float, paramètres déterminés par moindre carré

    Returns
    -------
    B : array

    """
    
    B = np.ones((len(x_obs)+1, 1))
    
    for i in range (len(x_obs)): 
        B[i] = vario_ajustee(x, y, x_obs[i], y_obs[i] , a0, C0)
                
    return B

#test2 = krig_grille(x_grd[0][0], y_grd[0][0], x_obs, y_obs, a0, C0)

def interp_kri( x_obs, y_obs, z_obs, x_grd, y_grd, a0, C0):
    """
    Interpolation par krigeage
    
    x_obs, y_obs, z_obs : vecteurs colonnes qui représentent les coordonnées des observations sur 
    lesquelles vont être effectuées l'interpolation
    
    x_grd, y_grd : arrays qui représentent les coordonnées planimétriques des points que l'on cherche à 
    interpoler
    
    Returns
    -------
    z_int : arrays donnant la valeur interpolée des points
    
    """
    A = krigeage (x_obs, y_obs, z_obs, a0, C0)
    z_int = np.nan*np.zeros(x_grd.shape)
    #z_incert = np.nan*np.zeros(x_grd.shape)
    
    for i in range (x_grd.shape[0]): 
        for j in range (x_grd.shape[1]): 
            
            B = krig_grille(x_grd[i,j], y_grd[i,j], x_obs, y_obs, a0, C0)
            res = np.linalg.solve(A, B)
            
            lam = res[:-1]
            mu = res[-1, 0]
            gi = B[:-1]
            
            z_int[i,j] = np.sum(lam*z_obs)
            #print(z_int[i,j])
            #z_incert[i,j] = np.sqrt(np.sum(lam*gi)+mu)
            
    gs.plot_surface_2d(x_grd, y_grd, z_int, x_obs, y_obs, xlabel = r'$x$ (m)', ylabel = r'$y$ (m)', title = r'Interpolation par krigeage (variogramme cubique)',minmax = [0,50])  

    #gs.plot_surface_2d(x_grd, y_grd, z_incert, x_obs, y_obs, xlabel = r'$x$ (m)', ylabel = r'$y$ (m)', 
                              #title = r'Interpolation par krigeage - incertitude (variogramme cubique)');
            
    return z_int

if __name__ == "__main__":
    
    #import des données 
#    data = np.loadtxt('points.dat')
#
#    x_obs = data[:,0:1]
#    y_obs = data[:,1:2]
#    z_obs = data[:,2:3]
#    
    survey = np.loadtxt('data_survey.txt')
    x_obs = survey[:,0:1]
    y_obs = survey[:,1:2]
    z_obs = survey[:,2:3]
    
    # ctrl = np.loadtxt('data_control.txt')
    # x_ctrl = ctrl[:,0:1]
    # y_ctrl = ctrl[:,1:2]
    # d_ctrl = ctrl[:,2:3]

    Hij, Dz = nuee_vario(x_obs, y_obs, z_obs)
    H_exp, Gam_exp = vario_exp(Hij, Dz, 500)
    # Gam_cub, a0, C0 = vario_cubique (Hij, Dz, H_exp, Gam_exp)
    # Gam_lin = gamma_lin(H_exp, Gam_exp)
    # x_grd, y_grd = np.meshgrid(np.linspace(np.floor(np.min(x_obs)), np.ceil(np.max(x_obs)), 100), np.linspace(np.floor(np.min(y_obs)), np.ceil(np.max(y_obs)), 100))
    # interp_krig = interp_kri(x_obs, y_obs, z_obs, a0, C0)
    
    