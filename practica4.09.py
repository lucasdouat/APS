#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thr Sep  4 20:39:19 

@author: lucasdouat
"""

#%% Mejorando las resoluciones espectrales con ventanas de Scipy Signal

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy.fft import fft

# Parámetros de la señal
fs = N = 1000

# Frecuencias de prueba
f1 = N / 4 * fs / N
f2 = (N / 4 + 1) * fs / N
f3 = (N / 4 + 0.175) * fs / N

# Tiempo
tt = np.linspace(0, (N - 1) / fs, N)

# Señales
x1 = np.sin(2 * np.pi * f1 * tt)
x2 = np.sin(2 * np.pi * f2 * tt)
x3 = np.sin(2 * np.pi * f3 * tt)

# Ventanas desde scipy.signal
ventana_rect = signal.windows.boxcar(N)
ventana_hamming = signal.windows.hamming(N)
ventana_blackmanharris = signal.windows.blackmanharris(N)

# Aplicar ventanas
x3_rect = x3 * ventana_rect
x3_hamming = x3 * ventana_hamming
x3_blackmanharris = x3 * ventana_blackmanharris

# FFT
X3_rect = fft(x3_rect)
X3_hamming = fft(x3_hamming)
X3_blackmanharris = fft(x3_blackmanharris)

# Frecuencia
df = fs / N
ll = np.linspace(0, ((N - 1) * df), N).flatten()
# Gráfico de espectros en magnitud
plt.figure(figsize=(10, 6))
plt.plot(ll, np.abs(X3_rect), '*', label='Rectangular')
plt.plot(ll, np.abs(X3_hamming), 'o', label='Hamming')
plt.plot(ll, np.abs(X3_blackmanharris),'.', label='Blackman-Harris')
plt.title('Espectro en Magnitud con Ventanas de scipy.signal')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud')
plt.grid(True)
plt.legend()
plt.xlim([0, fs / 2])
plt.tight_layout()
plt.show()

# Gráfico de espectros en dB
plt.figure(figsize=(10, 6))
plt.plot(ll, 20 * np.log10(np.abs(X3_rect)), '*', label='Rectangular')
plt.plot(ll, 20 * np.log10(np.abs(X3_hamming)), 'o', label='Hamming')
plt.plot(ll, 20 * np.log10(np.abs(X3_blackmanharris)),'.', label='Blackman-Harris')
plt.title('Espectro en dB con Ventanas de scipy.signal')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud (dB)')
plt.grid(True)
plt.legend()
plt.xlim([0, fs / 2])
plt.tight_layout()
plt.show()

