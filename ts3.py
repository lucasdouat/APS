#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 20:49:33 2025

@author: lucas-douat

Tarea Semanl N°3: Análisis de Fourier: FFT, desparramo, interpolación y ventaneo.
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import fft as fft

#Parámetros de la señal
fs = N = 1000 #fs: Frecnuencia de Muestro | N: Cantidad de Muestras.
df = fs/N # Resolución Espectral
ts = 1/fs # Resolución Temporal

#k0*(fs/N)
f1=N/4*df
f2=((N/4)+0.25)*df
f3=((N/4)+0.5)*df

#Sintetización de Señales:
tt=np.linspace(0, (N-1)*ts,N).flatten()
x1=np.sin(2*np.pi*f1*tt)
x2=np.sin(2*np.pi*f2*tt)
x3=np.sin(2*np.pi*f3*tt)

#Generación de Vectores para las FFT de mis señales.
X1=np.zeros(N,dtype=np.complex128())
X2=np.zeros(N,dtype=np.complex128())
X3=np.zeros(N,dtype=np.complex128())

#Generación del Vector de Frecuencia
ll= np.linspace(0, (N-1)*df,N).flatten()

#Calculo de la FFT mediante la función de numpy
X1=fft.fft(x1)
X2=fft.fft(x2)
X3=fft.fft(x3)

#Graficación de Espectro en Magnitud
plt.figure(1)
plt.clf()
plt.plot(ll,np.abs(X1),'x',label='|X1(f)|')
plt.plot(ll,np.abs(X2),'o',label='|X2(f)|')
plt.plot(ll,np.abs(X3),'*',label='|X3(f)|')
plt.title('Espectro en Magnitud')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('|X[k]|')
plt.grid(True)
plt.tight_layout()
plt.legend()#Muestra los labels de los plots
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



#%% d) Parseval

# Normalización para potencia unitaria (varianza = 1)
x1_norm = x1 / np.sqrt(np.var(x1))
x2_norm = x2 / np.sqrt(np.var(x2))
x3_norm = x3 / np.sqrt(np.var(x3))

# FFT de las señales normalizadas
X1 = fft.fft(x1_norm)
X2 = fft.fft(x2_norm)
X3 = fft.fft(x3_norm)

# Verificación de Parseval
energy_time_x1 = np.sum(x1_norm**2)
energy_freq_x1 = np.sum(np.abs(X1)**2) / N

energy_time_x2 = np.sum(x2_norm**2)
energy_freq_x2 = np.sum(np.abs(X2)**2) / N

energy_time_x3 = np.sum(x3_norm**2)
energy_freq_x3 = np.sum(np.abs(X3)**2) / N

# Resultados
print("x1 - Varianza:", np.var(x1_norm), "Energía temporal:", energy_time_x1, "Energía frecuencial:", energy_freq_x1)
print("x2 - Varianza:", np.var(x2_norm), "Energía temporal:", energy_time_x2, "Energía frecuencial:", energy_freq_x2)
print("x3 - Varianza:", np.var(x3_norm), "Energía temporal:", energy_time_x3, "Energía frecuencial:", energy_freq_x3)

#%% c) Aplicar Zero Padding
x1_zp = np.concatenate([x1, np.zeros(9*N)])
X1_zp = fft.fft(x1_zp)
x2_zp = np.concatenate([x2, np.zeros(9*N)])
X2_zp = fft.fft(x2_zp)
x3_zp = np.concatenate([x3, np.zeros(9*N)])
X3_zp = fft.fft(x3_zp)

#Nuevo Vector de Frecuencia
ll_zp = np.linspace(0, fs, len(X1_zp)).flatten()

# Graficas de Espectro en dB con Zeo Padding
plt.figure(3)
plt.clf()
plt.plot(ll_zp, 20*np.log10(np.abs(X1_zp)),'x',label='X1 dB')
plt.plot(ll_zp, 20*np.log10(np.abs(X2_zp)),'o',label='X2 dB')
plt.plot(ll_zp, 20*np.log10(np.abs(X3_zp)),'*',label='X3 dB')
plt.title("FFT con Zero Padding")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud (dB)")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.xlim([220, 280])
plt.show()
