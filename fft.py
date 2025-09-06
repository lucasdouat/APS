#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 20:49:33 2025

@author: lucas-douat
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import fft as fft

# Parámetros de la señal
fs = N = 1000
df = fs/N #Resolución espectral
ts = 1/fs #Resolución temporal

f1 = N/4*df #Frecuencia de (N/4)*df
f2 = ((N/4)+1)*df #Frecuencia = f1 + df
f3 = ((N/4)+0.175)*df


# Tiempo y señal
ts = 1 / fs
tt = np.linspace(0, (N - 1) * ts, N).flatten()
x1 = np.sin(2 * np.pi * f1 * tt)
x2 = np.sin(2* np.pi * f2 *tt)
x3 = np.sin(2 * np.pi * f3 * tt)


"""# Cálculo correcto de la DFT
X = np.zeros(N, dtype=np.complex128)
for k in range(N):
    for n in range(N):
        X[k] += x[n] * np.exp(-1j * 2 * np.pi * k * n / N)"""

X1 = np.zeros(N, dtype=np.complex128)
X2 = np.zeros(N, dtype=np.complex128)
X3 = np.zeros(N, dtype=np.complex128)


X1 = fft.fft(x1)
X2 = fft.fft(x2)
X3 = fft.fft(x3)



# Frecuencias
ll = np.linspace(0, ((N - 1) * df), N).flatten()

# Grafica de Especectro en Magnitud
plt.figure(1)
plt.clf()
plt.plot(ll,np.abs(X1), 'x', label = 'X1 abs')
plt.plot(ll,np.abs(X2), 'o', label = 'X2 abs')
plt.plot(ll,np.abs(X3), '*', label = 'X3 abs')
plt.title('FFT')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('|X1[k]|')
plt.grid(True)
plt.tight_layout()
plt.legend() #Muestra mis labels de la señal.
plt.xlim([0,fs/2])
plt.show()


# Grafica de Espectro en dB
plt.figure(2)
plt.clf()
plt.plot(ll, 20 * np.log10(np.abs(X1)) , 'x', label='X1 dB')
plt.plot(ll, 20 * np.log10(np.abs(X2)), 'o', label='X2 dB')
plt.plot(ll, 20 * np.log10(np.abs(X3)), '*', label='X3 dB')
plt.title('FFT en dB')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud (dB)')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.xlim([0, fs/2])
plt.show()


"""
# Graficar espectro X2
plt.figure(figsize=(8, 4))
plt.plot(ll, np.abs(X2))
plt.title('Espectro de Magnitud - X2')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('|X2[k]|')
plt.grid(True)
plt.tight_layout()
plt.show()

# Graficar espectro X3
plt.figure(figsize=(8, 4))
plt.plot(ll, np.abs(X3))
plt.title('Espectro de Magnitud - X3')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('|X3[k]|')
plt.grid(True)
plt.tight_layout()
plt.show()


plt.tight_layout()
plt.show()"""


