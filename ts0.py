#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on ...

@author: Lucas Douat

Descripción:
------------
Tarea Semanal N°0 - Primeros pasos en la simulación


Incluye:
- Función senoidal parametrizable
- Implementación de una señal triangular

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import sawtooth


#%%  Generación de señales

#Defino una función llamada mi_funcion_sen que acepta como parametros lo requerido en la consigna

def mi_funcion_sen(vmax, dc, ff, ph, nn, fs):
    tt = np.arange(nn) / fs # Generación del vector Tiempo desde 0 hasta el tiempo total de meustras
    xx = vmax*np.sin(2*np.pi*ff*tt+ph)+dc ##Generación de la señal senoidal usando los parametros y el vector tiempo.
    return tt.reshape(-1,1), xx.reshape(-1,1) #Retorno los vectores xx y tt, haciendo un ajuste en el formato "Vectores en colmunas Nx1    

#Funcion Triangular para Bonus

def mi_funcion_triangular(vmax, dc, ff, ph, nn, fs):
    tt = np.arange(nn) / fs
    xx = vmax * sawtooth(2 * np.pi * ff * tt + ph, width=0.5) + dc
    return tt.reshape(-1, 1), xx.reshape(-1, 1)

#%% Testbench

fs = 1000.0  # frecuencia de muestreo (Hz)
N = 1000     # cantidad de muestras

#Señal senoidal
tt, xx = mi_funcion_sen(vmax=1, dc=0.5, ff=5, ph=np.pi/4, nn=N, fs=fs)

plt.plot(tt, xx)
plt.title("Señal Senoidal Generada")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid(True)
plt.show()

#Bonus

tt_tri, xx_tri = mi_funcion_triangular(vmax=1, dc=0.5, ff=5, ph=np.pi/4, nn=N, fs=fs)
plt.figure()
plt.plot(tt_tri, xx_tri)
plt.title("Señal Triangular - 5 Hz")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid(True)
plt.show()




    

