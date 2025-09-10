#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 17:13:20 2025

@author: lucas-douat
"""

import numpy as np
import matplotlib.pyplot as plt
from ts1 import x1,x2,x3,x4,x5,x6
from scipy.signal import lfilter, square

#%%#%% 1) Ecuación en diferencia que modela un sistema LTI:

# Parámetros de simulación
fs = 100000  # frecuencia de muestreo en Hz
ts = 1 / fs  # tiempo entre muestras
N = 1100     # número de muestras
Npad = 30000 # cantidad de muestras con padding

# Nuevo vector de tiempo para graficar señales con padding
tt_padded = np.linspace(0, (Npad - 1) * ts, Npad)

# Coeficientes del sistema LTI
b = [0.03, 0.05, 0.03]  # coeficientes de entrada (numerador)
a = [1, -1.5, 0.5]      # coeficientes de salida (denominador)

# Aplicar lfilter con zero-padding a cada señal
señales = [x1, x2, x3, x4, x5, x6]
nombres = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']

print("Frecuencia de muestreo:", fs, "Hz")
print("Tiempo de simulación con padding:", round(Npad * ts, 4), "segundos")

for i, x in enumerate(señales):
    x_padded = np.zeros(Npad)
    x_padded[:N] = x
    y = lfilter(b, a, x_padded)
    potencia = np.mean(y**2)

    plt.figure()
    plt.plot(tt_padded, y)
    plt.title(f"Salida del sistema LTI con zero-padding para señal {nombres[i]}")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("y[n]")
    plt.grid(True)
    plt.tight_layout()
    print(f"{nombres[i]} | Potencia con padding: {round(potencia, 4)}")


#%% Sistema LTI  usando condiciones
def sistema_lti(x):
    N = len(x)
    y = np.zeros(2*N+(N-1))
    #y = np.zeros(N)
    
    for n in range(N):
        #Condiciones para acceder a posiciones validas de las listas.
        x_n   = x[n] if n >= 0 else 0
        x_n1  = x[n-1] if n-1 >= 0 else 0
        x_n2  = x[n-2] if n-2 >= 0 else 0
        y_n1  = y[n-1] if n-1 >= 0 else 0
        y_n2  = y[n-2] if n-2 >= 0 else 0
        
        y[n] = 0.03*x_n + 0.05*x_n1 + 0.03*x_n2 + 1.5*y_n1 - 0.5*y_n2
    
    return y

# Impulso Unitaroio
impulso = np.zeros(N)
impulso[0] = 1  # impulso unitario

h1 = sistema_lti(impulso)
tt2 = np.linspace(0, (2*N+(N-1))*ts,2*N+(N-1)).flatten()

# Graficar
plt.figure()
plt.plot(range(len(h1)),h1)
plt.title("Respuesta al impulso del sistema LTI mediante condiciones")
plt.xlabel("n")
plt.ylabel("h[n]")
plt.grid(True)
plt.show()

h2 = lfilter(b,a,impulso)

# Graficar
plt.figure()
plt.plot(range(len(h2)),h2)
plt.title("Respuesta al impulso del sistema LTI usando lfilter")
plt.xlabel("n")
plt.ylabel("h[n]")
plt.grid(True)
plt.show()

#%% Convolución con señales
señales = [x1, x2, x3, x4, x5, x6]
labels = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']

for i, x in enumerate(señales):
    y_conv = np.convolve(x, h1, mode='same')
    plt.figure()
    plt.plot(tt2,y_conv)
    plt.title(f"Salida por convolución con {labels[i]}")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("y[n]")
    plt.grid(True)
    plt.show()
    
#%% Sistema  y[n] = x[n] + 3x[n-10]
tt = np.linspace(0, (2*N+(N-1))*ts, N).flatten()
b1 = np.zeros(11)
b1[0] = 1
b1[10] = 3
a1 = [1]

h1 = lfilter(b1, a1, impulso)
plt.figure()
plt.plot(h1)
plt.title("Respuesta al impulso del sistema y[n] = x[n] + 3x[n-10]")
plt.xlabel("n")
plt.ylabel("h[n]")
plt.grid(True)

y1 = lfilter(b1, a1, x1)
plt.figure()
plt.plot(tt, y1)
plt.title("Salida del sistema con señal x1")
plt.xlabel("Tiempo [s]")
plt.ylabel("y[n]")
plt.grid(True)

#%% Sistema: y[n] = x[n] + 3y[n-10]
b2 = [1]
a2 = np.zeros(11)
a2[0] = 1
a2[10] = -3

h2 = lfilter(b2, a2, impulso)
plt.figure()
plt.plot(h2)
plt.title("Respuesta al impulso del sistema y[n] = x[n] + 3y[n-10]")
plt.xlabel("n")
plt.ylabel("h[n]")
plt.grid(True)

y2 = lfilter(b2, a2, x1)
plt.figure()
plt.plot(tt, y2)
plt.title("Salida del sistema con señal x1")
plt.xlabel("Tiempo [s]")
plt.ylabel("y[n]")
plt.grid(True)