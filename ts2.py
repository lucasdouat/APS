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
N =  1100    # número de muestras
tt = np.linspace(0, (N-1)*ts, N).flatten()


# Coeficientes del sistema LTI
# y[n] = 0.03*x[n] + 0.05*x[n-1] + 0.03*x[n-2] + 1.5*y[n-1] - 0.5*y[n-2]
b = [0.03, 0.05, 0.03]  # coeficientes de entrada (numerador)
a = [1, -1.5, 0.5]      # coeficientes de salida (denominador)

#Aplicar lfilter a cada señal
señales = [x1, x2, x3, x4, x5, x6]
nombres = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']

for i, x in enumerate(señales):
    y = lfilter(b, a, x)
    plt.figure()
    plt.plot(tt, y)
    plt.title(f"Salida del sistema LTI usando lfilter para señal {nombres[i]}")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("y[n]")
    plt.grid(True)
    plt.show()

# Sistema LTI  usando condiciones
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

y = lfilter(b, a,h)

# Graficar
plt.plot(y)
plt.title("Respuesta al impulso del sistema LTI usando lfilter")
plt.xlabel("n")
plt.ylabel("y[n]")
plt.grid(True)



#%% Convolución con señales
señales = [x1, x2, x3, x4, x5, x6]
labels = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']

for i, x in enumerate(señales):
    y_conv = np.convolve(x, h, mode='same')
    plt.figure()
    plt.plot(range(len(y_conv)), y_conv)
    plt.title(f"Salida por convolución con {labels[i]}")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("y[n]")
    plt.grid(True)
    plt.show()


#%% Markdown explicativo
"""
## 🧠 Importancia de la Cantidad de Muestras en Simulación de Sistemas LTI

La cantidad de muestras `N` tiene un impacto directo en la calidad de la simulación de señales y sistemas:

### 1. Mejor resolución temporal
- Aumentar `N` permite observar la evolución de la señal durante más tiempo.
- Es útil para sistemas con respuestas prolongadas (como los recursivos).

### 2. Mejor resolución espectral
- La resolución en frecuencia está dada por: Δf = fs / N
- Aumentar `N` reduce Δf, permitiendo distinguir mejor componentes cercanas en frecuencia.

### 3. Evitar efectos de borde
- En convolución, si `N` es pequeño, la salida puede estar truncada.
- Usar `mode='same'` ayuda, pero conviene tener margen suficiente.

### Recomendación
Para representar correctamente la respuesta de sistemas LTI, especialmente recursivos, se recomienda usar una cantidad de muestras suficientemente grande (por ejemplo, N = 10000 o más).
"""

#%%#%% Funciones auxiliares

def pot(x):
    return np.mean(x**2)

def energia(x):
    return np.sum(x**2)

#%% 1) Sintetizar señales
fs = 100000
Ts = 1 / fs
N = 1100
tt = np.linspace(0, (N-1)*Ts, N).flatten()

f1 = 2000
x1 = np.sin(2 * np.pi * f1 * tt)
x2 = 2 * np.sin(2 * np.pi * f1 * tt + np.pi/2)
f_mod = 1000
modulador = np.sin(2 * np.pi * f_mod * tt)
x3 = x1 * modulador
x4 = np.clip(x1, -0.75, 0.75)
f5 = 4000
x5 = square(2 * np.pi * f5 * tt)
N_pulso = int(0.01/Ts)
x6 = np.zeros(N)
x6[:N_pulso] = 1

#%% 2) Respuesta al impulso
impulso = np.zeros(N)
impulso[0] = 1
h = sistema_lti(impulso)

plt.figure()
plt.plot(h)
plt.title("Respuesta al impulso h[n]")
plt.xlabel("n")
plt.ylabel("h[n]")
plt.grid(True)
plt.tight_layout()
plt.show()

#%% 3) Convolución con señales x1 a x6
señales = [x1, x2, x3, x4, x5, x6]
nombres = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']

for i, x in enumerate(señales):
    y_conv = np.convolve(x, h, mode='same')
    plt.figure()
    plt.plot(range(len(y_conv)), y_conv)
    plt.title(f"Salida por convolución: {nombres[i]}")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("y[n]")
    plt.grid(True)
    plt.tight_layout()
    #energia = energia(y_conv)
    potencia = pot(y_conv)
    print(f"Salida {nombres[i]}  Potencia: {potencia:.4f}")

