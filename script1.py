#...start_script1...#
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# main python console for all scripts: https://trinket.io/embed/python3/3233e6f888?toggleCode=true&runOption=run&start=result

import matplotlib.pyplot as plt
import numpy as np
import time
import importlib
from scipy.interpolate import make_interp_spline


# Load Cu-Si data
# t=time/s, T = Temp/Kelvin, R_P_1 = R_Probe_1/Ohm (Cu), R_T = R_Thermometer/Ohm, R_P_2 = R_Probe_2/Ohm (Si)
t, T, R_P_1, R_T, R_P_2 = np.loadtxt("Heinzelmann_Vincent_Cu-Si.dat",  unpack = True, skiprows = 6)

#print(t, T, R_P_1, R_T, R_P_2)

# Plot T over t
fig = plt.figure(figsize=(8, 4), dpi=120).add_subplot(1, 1, 1)
plt.plot(t,T,'-', label='Temperature')
plt.xlabel(r"Time t / s")
plt.ylabel(r"Temperature T / K")
plt.legend()
plt.xlim(0, 5500)
plt.ylim(0, 320)
plt.title(r"Temperature over time of the Warm up process")
plt.show()


# Plot R_P_1 over T, (R_P_1 = R_Probe_1/Ohm)(Cu)
fig = plt.figure(figsize=(8, 4), dpi=120).add_subplot(1, 1, 1)
plt.plot(T,R_P_1,'-', label='Resistance of Cu')
plt.xlabel(r"Temperature T / K")
plt.ylabel(r"Resistance R / $\Omega$")
plt.legend()
plt.xlim(0, 320)
plt.ylim(0, 3)
plt.title(r"Resistance of Cu over Temperature")
plt.show()


# Plot R_T over T, (R_T = R_Thermometer/Ohm)
fig = plt.figure(figsize=(8, 4), dpi=120).add_subplot(1, 1, 1)
plt.plot(T,R_T,'-', label='Resistance of Thermometer 1')
plt.xlabel(r"Temperature T / K")
plt.ylabel(r"Resistance R / $\Omega$")
plt.legend()
plt.xlim(0, 320)
plt.ylim(0, 120)
plt.title(r"Resistance of Thermometer 1 over Temperature")
plt.show()


# Plot R_P_2 over T, (R_P_2 = R_Probe_2/Ohm (Si))
fig = plt.figure(figsize=(8, 4), dpi=120).add_subplot(1, 1, 1)
plt.plot(T,R_P_2,'-', label='Resistance of Si')
plt.xlabel(r"Temperature T / K")
plt.ylabel(r"Resistance R / $\Omega$")
plt.legend()
plt.xlim(0, 320)
#plt.ylim(0, 120)
#plt.yscale('log')
plt.title(r"Resistance of Si over Temperature")
plt.show()

# Load Nb-Si data
# t=time/s, T = Temp/Kelvin, R_P_1 = R_Probe_1/Ohm (Nb), R_T = R_Thermometer/Ohm, R_P_2 = R_Probe_2/Ohm (Si)
t, T, R_P_1, R_T, R_P_2 = np.loadtxt("Heinzelmann_Vincent_Nb-Si.dat",  unpack = True, skiprows = 6)

# Plot T over t
fig = plt.figure(figsize=(8, 4), dpi=120).add_subplot(1, 1, 1)
plt.plot(t,T,'-', label='Temperature')
plt.xlabel(r"time t/s")
plt.ylabel(r"Temperature T / K")
plt.legend()
plt.xlim(0, 4400)
plt.ylim(0, 300)
plt.title(r"Temperature over time of the cool down process")
plt.show()


# Plot R_P_1 over T, (R_P_1 = R_Probe_1/Ohm)(Nb)
fig = plt.figure(figsize=(8, 4), dpi=120).add_subplot(1, 1, 1)
plt.plot(T,R_P_1,'-', label='Resistance of Nb')
plt.xlabel(r"Temperature T / K")
plt.ylabel(r"Resistance R / $\Omega$")
plt.legend()
plt.xlim(0, 320)
#plt.ylim(0, 3)
plt.title(r"Resistance of Nb over Temperature")
plt.show()


# Plot R_P_2 over T, (R_P_2 = R_Probe_2/Ohm (Si))
fig = plt.figure(figsize=(8, 4), dpi=120).add_subplot(1, 1, 1)
plt.plot(T,R_P_2,'-', label='Resistance of Si')
plt.xlabel(r"Temperature T / K")
plt.ylabel(r"Resistance R / $\Omega$")
plt.legend()
plt.xlim(0, 320)
#plt.ylim(0, 120)
#plt.yscale('log')
plt.title(r"Resistance of Si over Temperature")
plt.show()


# Load Nb-H-Field data
# t=time/s, T = Temp/Kelvin, R_P_1 = R_Probe_1/Ohm (Nb), R_T = R_Thermometer/Ohm, R_P_2 = R_Probe_2/Ohm (Si)
t, T, R_P_1, R_T, R_P_2 = np.loadtxt("Heinzelmann_Vincent_Nb_H-Feld.dat",  unpack = True, skiprows = 8)

# Plot T over t
fig = plt.figure(figsize=(8, 4), dpi=120).add_subplot(1, 1, 1)
plt.plot(t[:790],T[:790],'-', label='Temperature')
plt.xlabel(r"time t/s")
plt.ylabel(r"Temperature T / K")
plt.legend()
plt.xlim(0, 1400)
plt.ylim(0, 12)
plt.title(r"Temperature over time with different H-Filed aplied")
plt.show()

# Plot T over t
fig = plt.figure(figsize=(8, 4), dpi=120).add_subplot(1, 1, 1)
plt.plot(t[790:],T[790:],'-', label='Temperature')
plt.xlabel(r"time t/s")
plt.ylabel(r"Temperature T / K")
plt.legend()
plt.xlim(0, 1400)
plt.ylim(0, 12)
plt.title(r"Temperature over time with different H-Filed aplied")
plt.show()


# Plot R_P_1 over T, (R_P_1 = R_Probe_1/Ohm)(Nb)
fig = plt.figure(figsize=(8, 4), dpi=120).add_subplot(1, 1, 1)
plt.plot(T[:625],R_P_1[:625],'.', label='Resistance of Nb warming up')
plt.plot(T[625:790],R_P_1[625:790],'.', label='Resistance of Nb cooling down')
plt.xlabel(r"Temperature T / K")
plt.ylabel(r"Resistance R / $\Omega$")
plt.legend()
#plt.xlim(0, 12)
plt.ylim(0.05, 0.075)
plt.title(r"Resistance of Nb over Temperature")
plt.show()

# Plot R_P_1 over T, (R_P_1 = R_Probe_1/Ohm)(Nb)
fig = plt.figure(figsize=(8, 4), dpi=120).add_subplot(1, 1, 1)
plt.plot(T[790:1185],R_P_1[790:1185],'.', label='Resistance of Nb warming up with H1')
plt.plot(T[1185:],R_P_1[1185:],'.', label='Resistance of Nb cooling down with H2')
plt.xlabel(r"Temperature T / K")
plt.ylabel(r"Resistance R / $\Omega$")
plt.legend()
#plt.xlim(0, 12)
plt.ylim(0.05, 0.075)
plt.title(r"Resistance of Nb over Temperature")
plt.show()


# Plot R_T over T, (R_T = R_Thermometer/Ohm)
fig = plt.figure(figsize=(8, 4), dpi=120).add_subplot(1, 1, 1)
plt.plot(T,R_T,'-', label='Resistance of Thermometer 1')
plt.xlabel(r"Temperature T / K")
plt.ylabel(r"Resistance R / $\Omega$")
plt.legend()
#plt.xlim(0, 12)
#plt.ylim(0.04, 0.08)
plt.title(r"Resistance of Thermometer 1 over Temperature")
plt.show()

#...end_script1...#