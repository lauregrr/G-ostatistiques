# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 15:37:29 2024

@author: Formation
"""

# Numpy
import numpy as np
# Matplotlib / plot
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay as delaunay
import math as m
import random
import lib_gs as gs
import tp_krigeage as k

    
from matplotlib import cm


#import de données pour tester les fonctions

survey = np.loadtxt('data_survey.txt')
x = survey[:,0:1]
y = survey[:,1:2]
d = survey[:,2:3]

ctrl = np.loadtxt('data_control.txt')
x_ctrl = ctrl[:,0:1]
y_ctrl = ctrl[:,1:2]
d_ctrl = ctrl[:,2:3]


#visualisation levé barymétrique - plot
# gs.plot_points(x, y, xlabel = '$x$ (m)', ylabel = '$y$ (m)', 
#               title = "sites barymétriques");

# Création d'une grille planimétrique pour l'interpolation
x_grd, y_grd = np.meshgrid(np.linspace(np.floor(np.min(x)), np.ceil(np.max(x)), 100), np.linspace(np.floor(np.min(y)), np.ceil(np.max(y)), 100))


#interpolation linéaire
#lin= gs.interp_lin(x, y, d)
#gs.plot_surface_2d(x_grd, y_grd, lin,x,y, xlabel = r'$x$ (m)', ylabel = r'$y$ (m)', 
#                title = r'Interpolation linéaire',minmax = [0,50]);
#EFFET DE BORD                   

#interpolation ppv
#ppv = gs.interp_ppv(x, y, d, x_grd, y_grd)
#gs.plot_surface_2d(x, y, ppv, x_grd, y_grd, xlabel = r'$x$ (m)', ylabel = r'y (m)', 
#                   title = r'Interpolation par plus proche voisin',minmax = [0,50]);

#interpolation par inverse des distances
#dist800= gs.interp_inv_distance(x, y, d, 2, 800)
#dist1600= gs.interp_inv_distance(x, y, d, 2, 1600)
#gs.plot_surface_2d(x_grd, y_grd, dist800, x, y, xlabel = r'$x$ (m)', ylabel = r'$y$ (m)', 
#                   title = r'Interpolation par inverse des distances ($p=2$, $d_{max}=800$)',minmax = [0,50]);
#gs.plot_surface_2d(x_grd, y_grd, dist1600, x, y, xlabel = r'$x$ (m)', ylabel = r'$y$ (m)', 
#                   title = r'Interpolation par inverse des distances ($p=2$, $d_{max}=1600$)',minmax = [0,50]);                 

#interpolation spline
#spline0 = gs.interp_spline(x, y, d, 0)
#spline1000 = gs.interp_spline(x, y, d, 1000)
#
#gs.plot_surface_2d(x_grd, y_grd, spline0, x, y, xlabel = r'$x$ (m)', ylabel = r'$y$ (m)',
#                   title = r'Interpolation par spline ($rho=0$)',minmax = [0,50]);
#gs.plot_surface_2d(x_grd, y_grd, spline1000, x, y, xlabel = r'$x$ (m)', ylabel = r'$y$ (m)',
#                   title = r'Interpolation par spline ($rho=1000$)',minmax = [0,50]);


#COMPARAISON AVEC NOS VALEURS DE BASE 

def comparaison (verif_d, test_d): 
    
    ecarts = test_d - verif_d
    mask = ~np.isnan(ecarts)
    E = ecarts[mask]
    verif_d_mask = verif_d[mask]
    moyenne_ref = np.mean(verif_d)
    
    mean = np.mean(E)
    std = np.std(E)
    rms = np.sqrt(np.mean(E**2))
    NSE = 1 - (np.sum(E**2)/np.sum((verif_d_mask-moyenne_ref)**2))
    
    res = np.array((mean, std, rms, NSE))

    return res 
    
if __name__ == "__main__":
    
    #Partie validation croisée 
    indices_points = list(range(len(x)))
    indices_aleatoires = random.sample(indices_points, 400)

    #liste données modèle
    model_x = []
    model_y = []
    model_d = []

    test_x = []
    test_y = []
    verif_d = []

    for i in range (len(x)): 
        if i in indices_aleatoires: 
            model_x.append([x[i][0]])
            model_y.append([y[i][0]])
            model_d.append([d[i][0]])
            
        else:
            test_x.append([x[i][0]])
            test_y.append([y[i][0]])
            verif_d.append([d[i][0]])
            
        
    model_x = np.array((model_x))
    model_y = np.array((model_y))
    model_d = np.array((model_d))

    test_x = np.array(test_x)
    test_y = np.array(test_y)
    verif_d = np.array(verif_d)
            
    #test avec le jeu modele
    #1- Interpolation linéaire 
    d_lin = gs.interp_lin(model_x, model_y, model_d, test_x, test_y)
    d_lin_comp = comparaison (verif_d, d_lin)

    #2- Distances inversées et distances inversées avec une distance max
    d_dist_inv, d_dist_max = gs.interp_inv_distance(model_x, model_y, model_d, test_x, test_y, 2, 1000)
    d_dist_inv_comp = comparaison (verif_d, d_dist_inv)
    d_dist_max_comp = comparaison (verif_d, d_dist_max)

    #3- Interpolation spline
    d_spline_rzero = gs.interp_spline(model_x, model_y, model_d, test_x, test_y, 0)
    d_spline_rmille = gs.interp_spline(model_x, model_y, model_d, test_x, test_y, 1000)
    d_spline_rzero_comp = comparaison (verif_d, d_spline_rzero)
    d_spline_rmille_comp = comparaison (verif_d, d_spline_rmille)

    #4- Interpolation krigeage
    Hij, Dz = k.nuee_vario(model_x, model_y, model_d)
    H_exp, Gam_exp = k.vario_exp(Hij, Dz, 500)
    Gam_cub, a0, C0 = k.vario_cubique (Hij, Dz, H_exp, Gam_exp)
    Gam_lin = k.gamma_lin(H_exp, Gam_exp)

    d_krig = k.interp_kri(model_x, model_y, model_d, test_x, test_y, a0, C0)
    d_krig_comp = comparaison (verif_d, d_krig)
    
    RES = np.array((d_lin_comp, d_dist_inv_comp, d_dist_max_comp, d_spline_rzero_comp, d_spline_rmille_comp, d_krig_comp))
    print(RES)
    
    #==> avec l'ensemble de ces paramètres, l'interpolation spline avec rho = 1000 semble être 
    # la meilleure méthode d'interpolation
    
    #Evaluation du levé avec les données de contrôle pour la méthode spline1000
    d_spline_rmille_final = gs.interp_spline(x, y, d, x_ctrl, y_ctrl, 1000)
    d_spline_rmille_controle = comparaison (d_ctrl, d_spline_rmille_final)
    


        


      
        


  
                   
                   
                   