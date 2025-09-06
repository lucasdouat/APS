#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on ...

@author: Lucas Douat

Descripción:
------------
Tarea Semanal N°1 - Síntesis de señales
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import square
import scipy.signal as sig



#%%  Definicion de Funciones
def pot(x):
    return np.mean(x**2)
    
def energia(x):
    return np.sum(x**2)


"""def ortogonalidad(x, y):
    producto_int = np.dot(x, y)
    return producto_int"""

#%% 1) Sintetizar y graficar:
fs = 100000  # frecuencia de muestreo en Hz
Ts = 1 / fs  # tiempo entre muestras
N = 1100     # número de muestras
tt = np.linspace(0, (N-1)*Ts, N).flatten() # vector de tiempo
"""
fs: frecuencia de muestreo 100 kHz.
Ts: tiempo entre muestras, calculado como el inverso de fs.
N: cantidad total de muestras.
tt: vector de tiempo, desde 0 hasta (N-1)*Ts.
"""
#Señal 1: Senoidal de 2 kHz
f1 = 2000
x1 = np.sin(2 * np.pi * f1 * tt)


#Señal 2: Amplificada y desfazada
x2 = 2 * np.sin(2 * np.pi * f1 * tt + np.pi/2)
"""
Misma señal que x1, pero:
amplitud duplicada (×2)
fase desplazada en pi/2 radianes
"""

#Señal 3: Modulada en amplitud
f_mod = 1000
modulador = np.sin(2 * np.pi * f_mod * tt)
x3 = x1 * modulador

"""Se genera una señal senoidal de 1 kHz (modulador)
Se usa como envolvente para modular la amplitud de x1
Esto simula una modulación AM (amplitud modulada)"""

#Señal 4: Recortada al 75% de la amplitud
x4 = np.clip(x1, -0.75,0.75)

# Señal 5: Cuadrada de 4 kHz
f5 = 4000
x5 = square(2 * np.pi * f5 * tt)


# Señal 6: Pulso rectangular de 10 ms
N_pulso = int(0.01/Ts)
x6 = np.zeros(N)
x6[:N_pulso] = 1

# Graficar todas las señales
fig, axs = plt.subplots(6, 1, figsize=(10, 10))
axs[0].plot(tt, x1)
axs[0].set_title(f"1) Senoidal 2 kHz | Ts={Ts:.2e}s | N={N} | Potencia={pot(x1):.3f}")

axs[1].plot(tt, x2)
axs[1].set_title(f"2) Amplificada y desfazada | Ts={Ts:.2e}s | N={N} | Potencia={pot(x2):.3f}")

axs[2].plot(tt, x3)
axs[2].set_title(f"3) Modul. AM por 1 kHz | Ts={Ts:.2e}s | N={N} | Potencia={pot(x3):.3f}")

axs[3].plot(tt, x4)
axs[3].set_title(f"4) Recortada al 75% Pot | Ts={Ts:.2e}s | N={N} | Potencia={pot(x4):.3f}")

axs[4].plot(tt, x5)
axs[4].set_title(f"5) Cuadrada 4 kHz | Ts={Ts:.2e}s | N={N} | Potencia={pot(x5):.3f}")

axs[5].plot(tt, x6)
axs[5].set_title(f"6) Pulso rectangular 10 ms | Ts={Ts:.2e}s | N={N} | Energia={energia(x6):.3f}")

for ax in axs:
    ax.set_xlim([0, N*Ts])
    ax.grid(True)

plt.tight_layout()
plt.ylabel("Amplitud [V]")
plt.show()

#%% 2) ortogonalidad entre la primera señal y las demás.

# Lista de señales y etiquetas
señales = [x2, x3, x4, x5, x6]
labels = ['x2', 'x3', 'x4', 'x5', 'x6']

# Función para calcular producto interno de x1 y las demas señales
def ortogonalidad(x1, señales):
    resultado = []
    for i, xi in enumerate(señales):
        producto_int = np.dot(x1, xi)
        resultado.append((labels[i], producto_int))
    return resultado

# Calcular ortogonalidad
resultado = ortogonalidad(x1, señales)

# Mostrar resultados
for label, dot in resultado:
    print(f"Producto interno entre x1 y {label}: {dot:.2f}")
    
#%% 3) Graficar la autocorrelación de la primera señal y la correlación entre ésta y las demás.

# Autocorrelación de x1 (Clase 21-08)
rxx = sig.correlate(x1, x1)

plt.figure("Autocorrelación rxx")
plt.plot(np.arange(-len(x1)+1, len(x1)),rxx)
plt.title("Autocorrelación rxx")
plt.xlabel("Índice")
plt.ylabel("Amplitud")
plt.grid(True)
plt.tight_layout()

# Función para calcular correlación cruzada
def correlacion(x1, señales):
    resultado = []
    for i, xi in enumerate(señales):
        xxi = sig.correlate(x1, xi, mode='full')
        resultado.append((labels[i], xxi))
    return resultado

# Calcular correlaciones cruzadas
resultado = correlacion(x1, señales)
lags = np.arange(-len(x1)+1, len(x1))

# Graficar cada correlación cruzada en una figura individual
for label, corr in resultado:
    plt.figure(figsize=(10, 4))
    plt.plot(lags, corr)
    plt.title(f"Correlación cruzada entre x1 y {label}")
    plt.xlabel("Desplazamiento")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



