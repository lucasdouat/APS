#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 20:20:25 2025

@author: lucas-douat
"""


import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

# Crear la señal x
M = 8
x = np.zeros(M)
x[3:5] = 1
print("Señal x:", x)

# Graficar la señal x usando stem en una figura separada
plt.figure("Señal x - Stem")
plt.stem(x)
plt.title("Señal x (stem)")
plt.xlabel("Índice")
plt.ylabel("Amplitud")
#plt.grid(True)

# Calcular la autocorrelación
rxx = sig.correlate(x, x)
print("Autocorrelación rxx:", rxx)

# Graficar la autocorrelación usando stem en otra figura
plt.figure("Autocorrelación rxx - Stem")
plt.stem(rxx)
plt.title("Autocorrelación rxx (stem)")
plt.xlabel("Índice")
plt.ylabel("Amplitud")
#plt.grid(True)

# Mostrar los gráficos
plt.show()
