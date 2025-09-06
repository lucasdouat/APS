#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 20:49:33 2025

@author: lucas-douat
"""

import numpy as np
import matplotlib.pyplot as plt

# Parámetros de la señal
fs = N = 1000
f1 = 1

# Tiempo y señal
ts = 1 / fs
tt = np.linspace(0, (N - 1) * ts, N).flatten()
x = np.sin(2 * np.pi * f1 * tt)

# Resolución espectral
df = fs / N

# Cálculo correcto de la DFT
X = np.zeros(N, dtype=np.complex128)
for k in range(N):
    for n in range(N):
        X[k] += x[n] * np.exp(-1j * 2 * np.pi * k * n / N)

# Frecuencias
ll = np.linspace(0, (N - 1) * df, N).flatten()

# Graficar espectro de magnitud y fase
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(ll, np.abs(X))
plt.title('Espectro de Magnitud')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('|X[k]|')

plt.subplot(2, 1, 2)
plt.plot(ll, np.angle(X))
plt.title('Espectro de Fase')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Fase de X[k]')

plt.tight_layout()
plt.show()
