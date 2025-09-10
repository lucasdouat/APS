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

# Parámetros de simulación
fs = 100000  # frecuencia de muestreo en Hz
Ts = 1 / fs  # tiempo entre muestras
N = 1100     # número de muestras
tt = np.linspace(0, (2*N+(N-1))*Ts, N).flatten()

#%% Señales de entrada importada de ts1

señales = [x1, x2, x3, x4, x5, x6]
nombres = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']

#%% Sistema LTI usando lfilter
#y[n] = 0.03*x[n] + 0.05*x[n-1] + 0.03*x[n-2] + 1.5*y[n-1] - 0.5*y[n-2]
b = [0.03, 0.05, 0.03]  # coeficientes de entrada (numerador)
a = [1, -1.5, 0.5]      # coeficientes de salida (denominador)


#%% Sistema LTI principal usando lfilter
b = [0.03, 0.05, 0.03]
a = [1, -1.5, 0.5]

print("Frecuencia de muestreo:", fs, "Hz")
print("Tiempo de simulación:", round(N * ts, 4), "segundos")

for i, x in enumerate(señales):
    y = lfilter(b, a, x)
    energia = np.sum(y**2)
    potencia = np.mean(y**2)
    plt.figure()
    plt.plot(tt, y)
    plt.title(f"Salida del sistema LTI para señal {nombres[i]}")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("y[n]")
    plt.grid(True)
    plt.tight_layout()
    print(f"Señal {nombres[i]} | Potencia: {potencia:.4f}")

# Respuesta al impulso
impulso = np.zeros(N)
impulso[0] = 1
h = lfilter(b, a, impulso)
plt.figure()
plt.plot(h)
plt.title("Respuesta al impulso del sistema LTI")
plt.xlabel("n")
plt.ylabel("h[n]")
plt.grid(True)
plt.tight_layout()

# Convolución con respuesta al impulso
y_conv = np.convolve(x3, h)
plt.plot(y_conv)
plt.title("Salida por convolución con h[n] usando x3")
plt.xlabel("n")
plt.ylabel("y[n]")
plt.grid(True)
plt.tight_layout()
potencia_conv = np.mean(y_conv**2)
print(f"Convolución con x3 | Potencia: {potencia_conv:.4f}")

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

h = sistema_lti(impulso)

# Graficar
plt.plot(range(len(h)),h)
plt.title("Respuesta al impulso del sistema LTI mediante condiciones")
plt.xlabel("n")
plt.ylabel("h[n]")
plt.grid(True)
plt.show()

#%% Sistema  y[n] = x[n] + 3x[n-10]
b_fir = np.zeros(11)
b_fir[0] = 1
b_fir[10] = 3
a_fir = [1]

h_fir = lfilter(b_fir, a_fir, impulso)
plt.figure()
plt.plot(h_fir)
plt.title("Respuesta al impulso del sistema y[n] = x[n] + 3x[n-10]")
plt.xlabel("n")
plt.ylabel("h[n]")
plt.grid(True)

y_fir = lfilter(b_fir, a_fir, x1)
plt.figure()
plt.plot(tt, y_fir)
plt.title("Salida del sistema con señal x1")
plt.xlabel("Tiempo [s]")
plt.ylabel("y[n]")
plt.grid(True)

#%% Sistema: y[n] = x[n] + 3y[n-10]
b_iir = [1]
a_iir = np.zeros(11)
a_iir[0] = 1
a_iir[10] = -3

h_iir = lfilter(b_iir, a_iir, impulso)
plt.figure()
plt.plot(h_iir)
plt.title("Respuesta al impulso del sistema y[n] = x[n] + 3y[n-10]")
plt.xlabel("n")
plt.ylabel("h[n]")
plt.grid(True)

y_iir = lfilter(b_iir, a_iir, x1)
plt.figure()
plt.plot(tt, y_iir)
plt.title("Salida del sistema con señal x1")
plt.xlabel("Tiempo [s]")
plt.ylabel("y[n]")
plt.grid(True)
