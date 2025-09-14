#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 17:13:20 2025

@author: lucas-douat
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, square

#%%#%% 1) Ecuación en diferencia que modela un sistema LTI:

# Parámetros de simulación
fs = 100000  # frecuencia de muestreo en Hz
ts = 1 / fs  # tiempo entre muestras
N = 1000     # número de muestras
tt = np.linspace(0, (N-1)*ts,N).flatten() # vector de tiempo.

# Nuevo vector de tiempo para graficar señales con padding
Npad = 2*N+(N-1) # cantidad de muestras con padding
tt_padded = np.linspace(0, (Npad - 1) * ts, Npad)

#Señal 1: Senoidal de 2kHz
f1 = 2000
x1 = np.sin(2*np.pi*f1*tt)

#Señal 2: Amplificada y desfazada
x2 = 2*np.sin(2*np.pi*f1*tt+np.pi/2)

#señal 3: Modulada en amplitud
f_mod = 1000
modulador = np.sin(2*np.pi*f_mod*tt)
x3=x1*modulador

#Señal 4: Recortada al 75% de la amplitud
x4 = np.clip(x1, -0.75,0.75)

#Señal 5. Cuadrada de 4kHz
f5 = 4000
x5 = square(2*np.pi*f5*tt)

#señal 6: Pulso rectangular de 10ms
N_pulso = int(0.01/ts)
x6= np.zeros(N)
x6[:N_pulso]=1

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
    print(f"{nombres[i]} | Potencia: {round(potencia, 4)}")


#%% Sistema LTI  usando condiciones
def sistema_lti(x):
    N = len(x)
    y = np.zeros(2*N+(N-1))
    
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

for i, x in enumerate(señales):
    x_padded = np.zeros(Npad)
    x_padded[:N] = x
    y_conv = np.convolve(x_padded, h1, mode='valid')
    plt.figure()
    plt.plot(y_conv)
    plt.title(f"Salida por convolución con {nombres[i]}")
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