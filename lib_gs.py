#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##########################################################################
#    TP - Introduction à l'interpolation spatiale et aux géostatistiques #
##########################################################################

# P. Bosser / ENSTA Bretagne
# Version du 26/02/2024


# Numpy
import numpy as np
# Matplotlib / plot
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay as delaunay
import math as m
    
from matplotlib import cm

#import de données pour tester les fonctions
data = np.loadtxt('points.dat')

x_obs = data[:,0:1]
y_obs = data[:,1:2]
z_obs = data[:,2:3]

################## Modèle de fonction d'interpolation ##################

def interp_xxx(x_obs, y_obs, z_obs, x_int, y_int):
    # Interpolation par ???
    # x_obs, y_obs, z_obs : observations
    # [np.array dimension 1*n]
    # x_int, y_int, positions pour lesquelles on souhaite interpoler une valeur z_int
    # [np array dimension m*p]
    
    z_int = np.nan*np.zeros(x_int.shape)
    #
    # ...
    #
    return z_int

####################### Fonctions d'interpolation ######################

def interp_lin(x_obs, y_obs, z_obs, x_grd, y_grd):
    """
    Fonction d'interpolation linaire à partir d'une triangulation de Delauney
    
    Parameters
    ----------
    x_obs, y_obs, z_obs : vecteurs colonnes qui représentent les coordonnées des observations sur 
    lesquelles vont être effectuées l'interpolation
    
    x_grd, y_grd : arrays qui représentent les coordonnées planimétriques des points que l'on cherche à 
    interpoler

    Returns
    -------
    z_int : arrays donnant la valeur interpolée des points

    """
    # On construit la triangulation :
    # tri est un tableau de 3 colonnes, le nombre de ligne correspond au nombres de triangles
    tri = delaunay( np.hstack( (x_obs, y_obs) ) )

    # Création de notre array vide pour stocker les valeurs interpolées
    z_int = np.nan*np.zeros(x_grd.shape)

    #boucle pour calculer chaque valeur à interpoler
    for i in range (np.shape(x_grd)[0]):
        for j in range (np.shape(x_grd)[1]):
            x0=x_grd[i][j]
            y0=y_grd[i][j]
            
            # on recherche le numéro du triangle dans tri contenant le point (x0,y0)
            idx_t = tri.find_simplex( np.array([x0, y0]) )
            
            # test pour voir si point de la grille appartient bien à un triangle
            # si le point n'appartient pas à un triangle, on passe à l'itération suivante 
            # et le point ne possède pas de valeur interpolée
            if idx_t==-1: 
                pass
            
            else: 
                
                # on récupère les numéros des sommets du triangle contenant le point (x0,y0)
                idx_s = tri.simplices[idx_t,:]
                
                # x_obs, y_obs sont des tableaux à 2 dimensions ; il faut les préciser pour en extraire un scalaire
                x1 = x_obs[ idx_s[0],0 ] ; y1 = y_obs[ idx_s[0],0 ] ; z1 = z_obs[ idx_s[0],0 ]
                x2 = x_obs[ idx_s[1],0 ] ; y2 = y_obs[ idx_s[1],0 ] ; z2 = z_obs[ idx_s[1],0 ]
                x3 = x_obs[ idx_s[2],0 ] ; y3 = y_obs[ idx_s[2],0 ] ; z3 = z_obs[ idx_s[2],0 ]
                
                A= np.array(((x1,y1,1), (x2,y2,1), (x3,y3,1)))
                B= np.array((z1,z2,z3))
                X= np.linalg.solve(A, B)
                
                z_int[i][j]=X[0]*x0+X[1]*y0+X[2]
                
    return z_int


#    plot_surface_2d(x_grd, y_grd, z_int, x_obs, y_obs, xlabel = r'$x$ (m)', ylabel = r'$y$ (m)', 
#                       title = r'Interpolation linéaire',minmax = [450,750]);
    
def interp_ppv(x_obs, y_obs, z_obs, x_int, y_int):
    # Interpolation par plus proche voisin
    # x_obs, y_obs, z_obs : observations
    # [np.array dimension 1*n]
    # x_int, y_int, positions pour lesquelles on souhaite interpoler une valeur z_int
    # [np array dimension m*p]
    
    z_int = np.nan*np.zeros(x_int.shape)
    for i in np.arange(0,x_int.shape[0]):
        for j in np.arange(0,x_int.shape[1]):
            d = np.sqrt((x_int[i,j]-x_obs)**2+(y_int[i,j]-y_obs)**2)
            idx = np.argmin(d)
            z_int[i,j] = z_obs[idx]
            
    return z_int

def interp_inv_distance(x_obs, y_obs, z_obs, x_grd, y_grd, p, dist_max):
    """
    Fonction d'interpolation qui réalise deux interpolations en même temps: 
        - interpolation par inverse de distances 
        - interpolation par inverse de distances avec une distance maximale
        
    Parameters
    ----------
    x_obs, y_obs, z_obs : vecteurs colonnes qui représentent les coordonnées des observations sur 
    lesquelles vont être effectuées l'interpolation
    
    x_grd, y_grd : arrays qui représentent les coordonnées planimétriques des points que l'on cherche à 
    interpoler
    
    p : integer
    
    dist_max : 

    Returns
    -------
    z_int : arrays donnant la valeur interpolée des points par inverse de distance 
    z_int_min : arrays donnant la valeur interpolée des points par inverse de distance avec valeur maximal  
    

    """
    
    # Création de notre array vide pour stocker les valeurs interpolées
    z_int = np.nan*np.zeros(x_grd.shape)
    z_int_min = np.nan*np.zeros(x_grd.shape)

    # Boucles pour compléter chaque valeur interpolée
    for i in range (np.shape(x_grd)[0]):
        for j in range (np.shape(x_grd)[1]):
            
            # Calcul de la distance entre le point à interpolée et 
            # l'ensemble de nos observations
            d = np.sqrt((x_grd[i,j]-x_obs)**2+(y_grd[i,j]-y_obs)**2)
            
            z_int[i,j]=np.sum(z_obs/d**p)/np.sum(1/d**p)
            
            z_int_min[i][j]=np.sum(z_obs[d<=dist_max]/d[d<=dist_max]**p)/np.sum(1/d[d<=dist_max]**p)
            
            
         
#    plot_surface_2d(x_grd, y_grd, z_int, x_obs, y_obs, xlabel = r'$x$ (m)', ylabel = r'$y$ (m)', 
#                        title = r'Interpolation par inverse des distances ($p=2$)',minmax = [450,750]);
#
#    plot_surface_2d(x_grd, y_grd, z_int_min, x_obs, y_obs, xlabel = r'$x$ (m)', ylabel = r'$y$ (m)', 
#                        title = r'Interpolation par inverse des distances ($p=2$) et d=30',minmax = [450,750]);
    
    return z_int, z_int_min

def interp_spline(x_obs, y_obs, z_obs, x_grd, y_grd, rho):
    """
    Fonction d'interpolation par spline d'interpolation ou lissage en fonction de la valeur de rho
    
    Parameters
    ----------
    x_obs, y_obs, z_obs : vecteurs colonnes qui représentent les coordonnées des observations sur 
    lesquelles vont être effectuées l'interpolation
    
    x_grd, y_grd : arrays qui représentent les coordonnées planimétriques des points que l'on cherche à 
    interpoler
    
    rho: interger, paramètre de l'interpolation par spline

    Returns
    -------
    z_int : arrays donnant la valeur interpolée des points

    """
    
    # Création de notre array vide pour stocker les valeurs interpolées
    z_int = np.nan*np.zeros(x_grd.shape)

    # Créer une ligne de 1
    ligne_de_1 = np.ones((1, len(x_obs)))  # Supposons que la longueur des vecteurs est la même

    # Concaténer les vecteurs et la ligne de 1 pour former une matrice
    D = np.concatenate((ligne_de_1, x_obs.reshape(1, -1), y_obs.reshape(1, -1)), axis=0)
    A = np.transpose(D)
    C = np.zeros((3,3))

    B = np.zeros((len(x_obs), len(x_obs)))

    for i in range (len(x_obs)):
        for j in range (len(y_obs)): 
            
            if i==j:
                B[i,j]=rho
                
            else: 
                d = np.sqrt((x_obs[i]-x_obs[j])**2+(y_obs[i]-y_obs[j])**2)
                B[i,j] = phi(d)
                
                
    # Créer la matrice en assemblant les blocs
    matrice = np.block([[A, B],
                        [C, D]])

    # Concaténer les deux vecteurs colonnes pour obtenir un seul vecteur colonne
    zero = np.zeros((3, 1))
    Z = np.vstack((z_obs, zero))

    X= np.linalg.solve(matrice, Z)
    bi = X[3:]


    for i in range (np.shape(x_grd)[0]):
        for j in range (np.shape(x_grd)[1]):
            
            vect1 = np.array(([1], [x_grd[i,j]], [y_grd[i,j]]))
            vect2 = np.sqrt((x_grd[i,j]-x_obs)**2+(y_grd[i,j]-y_obs)**2)
            resultat_phi = np.vectorize(phi)(vect2)
            
            calc = np.vstack((vect1, resultat_phi))
            
            z_int[i,j] = np.transpose(calc)@X
            
#    plot_surface_2d(x_grd, y_grd, z_int, x_obs, y_obs, xlabel = r'$x$ (m)', ylabel = r'$y$ (m)',  
#                       title = r'Interpolation par spline ($rho=0$)',minmax = [450,750]);
    return z_int

############################# Fonction intermédiaire dont on peut avoir besoin ############################
def phi(r):
    """
    Parameters
    ----------
    r : Float

    Returns: float
    """
    phi= (r**2)*m.log(r)
    
    return phi

############################# Visualisation ############################

def plot_contour_2d(x_grd ,y_grd ,z_grd, x_obs = np.array([]) ,y_obs = np.array([]), xlabel = "", ylabel = "", title = "", fileo = ""):
    # Tracé du champ interpolé sous forme d'isolignes
    # x_grd, y_grd, z_grd : grille de valeurs interpolées
    # x_obs, y_obs : observations (facultatif)
    # xlabel, ylabel : étiquettes des axes (facultatif)
    # title : titre (facultatif)
    # fileo : nom du fichier d'enregistrement de la figure (facultatif)
    
    z_grd_m = np.ma.masked_invalid(z_grd)
    fig = plt.figure()
    plt.contour(x_grd, y_grd, z_grd_m, int(np.round((np.max(z_grd_m)-np.min(z_grd_m))/4)),colors ='k')
    if x_obs.shape[0]>0:
        plt.scatter(x_obs, y_obs, marker = 'o', c = 'k', s = 5)
        dx = max(x_obs)-min(x_obs)
        dy = max(y_obs)-min(y_obs)
        minx = min(x_obs)-0.05*dx; maxx = max(x_obs)+0.05*dx
        miny = min(y_obs)-0.05*dy; maxy = max(y_obs)+0.05*dy
    else:
        dx = np.max(x_grd)-np.min(x_grd)
        dy = np.max(y_grd)-np.min(y_grd)
        minx = np.min(x_grd)-0.05*dx; maxx = np.max(x_grd)+0.05*dx
        miny = np.min(y_grd)-0.05*dy; maxy = np.max(y_grd)+0.05*dy
    plt.xlim([minx,maxx])
    plt.ylim([miny,maxy])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    if not fileo == "": plt.savefig(fileo,bbox_inches='tight')
    return plt.gca()

def plot_surface_2d(x_grd ,y_grd ,z_grd, x_obs = np.array([]) ,y_obs = np.array([]), minmax = [0,0], xlabel = "", ylabel = "", zlabel = "", title = "", fileo = "", cmap = cm.terrain):
    # Tracé du champ interpolé sous forme d'une surface colorée
    # x_grd, y_grd, z_grd : grille de valeurs interpolées
    # x_obs, y_obs : observations (facultatif)
    # minmax : valeurs min et max de la variable interpolée (facultatif)
    # xlabel, ylabel, zlabel : étiquettes des axes (facultatif)
    # title : titre (facultatif)
    # fileo : nom du fichier d'enregistrement de la figure (facultatif)
    # cmap : nom de la carte de couleur
    z_grd_m = np.ma.masked_invalid(z_grd)
    fig = plt.figure()
    if minmax[0] < minmax[-1]:
        p=plt.pcolormesh(x_grd, y_grd, z_grd_m, cmap=cmap, vmin = minmax[0], vmax = minmax[-1], shading = 'auto')
    else:
        p=plt.pcolormesh(x_grd, y_grd, z_grd_m, cmap=cmap, shading = 'auto')
    if x_obs.shape[0]>0:
        plt.scatter(x_obs, y_obs, marker = 'o', c = 'k', s = 5)
        dx = max(x_obs)-min(x_obs)
        dy = max(y_obs)-min(y_obs)
        minx = min(x_obs)-0.05*dx; maxx = max(x_obs)+0.05*dx
        miny = min(y_obs)-0.05*dy; maxy = max(y_obs)+0.05*dy
    else:
        dx = np.max(x_grd)-np.min(x_grd)
        dy = np.max(y_grd)-np.min(y_grd)
        minx = np.min(x_grd)-0.05*dx; maxx = np.max(x_grd)+0.05*dx
        miny = np.min(y_grd)-0.05*dy; maxy = np.max(y_grd)+0.05*dy
    plt.xlim([minx,maxx])
    plt.ylim([miny,maxy])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    fig.colorbar(p,ax=plt.gca(),label=zlabel,fraction=0.046, pad=0.04)
    if not fileo == "": plt.savefig(fileo,bbox_inches='tight')
    return plt.gca()

def plot_points(x_obs, y_obs, xlabel = "", ylabel = "", title = "", fileo = ""):
    # Tracé des sites d'observations
    # x_obs, y_obs : observations
    # xlabel, ylabel : étiquettes des axes (facultatif)
    # title : titre (facultatif)
    # fileo : nom du fichier d'enregistrement de la figure (facultatif)
    
    fig = plt.figure()
    ax = plt.gca()
    plt.plot(x_obs, y_obs, 'ok', ms = 4)
    dx = max(x_obs)-min(x_obs)
    dy = max(y_obs)-min(y_obs)
    minx = min(x_obs)-0.05*dx; maxx = max(x_obs)+0.05*dx
    miny = min(y_obs)-0.05*dy; maxy = max(y_obs)+0.05*dy
    ax.set_aspect('equal', adjustable='box')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    if not fileo == "": plt.savefig(fileo,bbox_inches='tight')
    return plt.gca()

def plot_patch(x_obs, y_obs, z_obs, xlabel = "", ylabel = "", zlabel = "", title = "", fileo = "", cmap = cm.terrain):
    # Tracé des valeurs observées
    # x_obs, y_obs, z_obs : observations
    # xlabel, ylabel, zlabel : étiquettes des axes (facultatif)
    # title : titre (facultatif)
    # fileo : nom du fichier d'enregistrement de la figure (facultatif)
    # cmap : nom de la carte de couleur
    
    fig = plt.figure()
    p=plt.scatter(x_obs, y_obs, marker = 'o', c = z_obs, s = 80, cmap=cmap)
    dx = max(x_obs)-min(x_obs)
    dy = max(y_obs)-min(y_obs)
    minx = min(x_obs)-0.05*dx; maxx = max(x_obs)+0.05*dx
    miny = min(y_obs)-0.05*dy; maxy = max(y_obs)+0.05*dy
    plt.xlim([minx,maxx])
    plt.ylim([miny,maxy])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    fig.colorbar(p,ax=plt.gca(),label=zlabel,fraction=0.046, pad=0.04)
    if not fileo == "": plt.savefig(fileo,bbox_inches='tight')
    return plt.gca()

def plot_triangulation(x_obs, y_obs, xlabel = "", ylabel = "", title = "", fileo = ""):
    # Tracé de la triangulation sur des sites d'observations
    # x_obs, y_obs : observations
    # xlabel, ylabel : étiquettes des axes (facultatif)
    # title : titre (facultatif)
    # fileo : nom du fichier d'enregistrement de la figure (facultatif)
    from scipy.spatial import Delaunay as delaunay
    tri = delaunay(np.hstack((x_obs,y_obs)))
    
    plt.figure()
    plt.triplot(x_obs[:,0], y_obs[:,0], tri.simplices)
    plt.plot(x_obs, y_obs, 'or', ms=4)
    dx = max(x_obs)-min(x_obs)
    dy = max(y_obs)-min(y_obs)
    minx = min(x_obs)-0.05*dx; maxx = max(x_obs)+0.05*dx
    miny = min(y_obs)-0.05*dy; maxy = max(y_obs)+0.05*dy
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    if not fileo == "": plt.savefig(fileo,bbox_inches='tight')
    return plt.gca()
