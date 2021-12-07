#...start_script1...#
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# main python console for all scripts: https://trinket.io/embed/python3/5cdac64636?toggleCode=true&runOption=run&start=result

import matplotlib.pyplot as plt
import numpy as np
import time
import importlib
from scipy.optimize import *
from scipy.interpolate import make_interp_spline


def exp(x,C,a,b,d):
    return C*np.exp(a*(x+b))+d

def lin(x,m,n):
    return m*x+n

def func1(x,a,b,c):
    return a*(x+b)**3 + c

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



# fitting the T^3 function
plot_range = [0,1087]
fit_range = [0,880]

fit_parameters = [["a" ,"b", "c"],   
                  [ 1e-5,     -0.01, 0.1],      # max bounds
                  [2e-6,   -4.2, 0.07],    # start values
                  [1e-6, -10, 0]]     # min bounds


popt, pcov = curve_fit(func1, T[fit_range[0]:fit_range[1]], R_P_1[fit_range[0]:fit_range[1]], fit_parameters[2], bounds=(fit_parameters[3],fit_parameters[1]))

opt_fit_parameters1 = popt.copy()
pcov1 = pcov.copy()
K_1=func1(4.2, popt[0], popt[1], popt[2])
print(popt)
print("Fit eq: y= a*(x+b)^3")
print("a= {:.4g} +/- {:.4g}, b= {:.4g} +/- {:.4g}, c= {:.4g} +/- {:.4g}".format(opt_fit_parameters1[0], np.sqrt(np.diag(pcov))[0], opt_fit_parameters1[1], np.sqrt(np.diag(pcov))[1], opt_fit_parameters1[2], np.sqrt(np.diag(pcov))[2]))
print("R(4.2K)= {:.4g}".format(K_1))

# Plot R_P_1 over T, (R_P_1 = R_Probe_1/Ohm)(Cu) reduced range with T^3 fit
fig = plt.figure(figsize=(8, 4), dpi=120).add_subplot(1, 1, 1)
plt.plot(T[:plot_range[1]],R_P_1[:plot_range[1]],'.', label='Resistance of Cu')
plt.plot(T[fit_range[0]:fit_range[1]], func1(T[fit_range[0]:fit_range[1]], *popt), 'r--', label="Fit: R= a*(T+b)^3")

plt.xlabel(r"Temperature T / K")
plt.ylabel(r"Resistance R / $\Omega$")
plt.legend()
#plt.xlim(0, 50)
#plt.ylim(0, 3)
plt.title(r"Resistance of Cu over Temperature")
plt.show()


# fitting the linear T function
plot_range = [2647,3850]
fit_range = [2647,3800]

fit_parameters = [["m" ,"n"],   
                  [   0.1,  0.0],      # max bounds
                  [  0.01,  -0.5],    # start values
                  [ 0.001,  -2]]     # min bounds


popt, pcov = curve_fit(lin, T[fit_range[0]:fit_range[1]], R_P_1[fit_range[0]:fit_range[1]], fit_parameters[2], bounds=(fit_parameters[3],fit_parameters[1]))

opt_fit_parameters2 = popt.copy()
pcov2 = pcov.copy()
#K_2=func1(300, popt[0], popt[1])
#print(popt)
print("Fit eq: y= m*x+n")
print("m= {:.4g} +/- {:.4g}, n= {:.4g} +/- {:.4g}".format(opt_fit_parameters1[0], np.sqrt(np.diag(pcov))[0], opt_fit_parameters1[1], np.sqrt(np.diag(pcov))[1]))
#print("R(4.2K)= {:.4g}".format(K_1))

# Plot R_P_1 over T, (R_P_1 = R_Probe_1/Ohm)(Cu) reduced range with T^1 fit
fig = plt.figure(figsize=(8, 4), dpi=120).add_subplot(1, 1, 1)
plt.plot(T,R_P_1,'.', label='Resistance of Cu')
plt.plot(T[fit_range[0]:fit_range[1]], lin(T[fit_range[0]:fit_range[1]], *popt), 'r--', label="Fit: R= m*T + n")

plt.xlabel(r"Temperature T / K")
plt.ylabel(r"Resistance R / $\Omega$")
plt.legend()
#plt.xlim(0, 50)
#plt.ylim(0, 3)
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
plt.plot(T,R_P_2,'.', label='Resistance of Si')
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
plt.xlim(0, 800)
plt.ylim(0, 12)
plt.title(r"Temperature over time with different H-Filed aplied")
plt.show()

# Plot T over t
fig = plt.figure(figsize=(8, 4), dpi=120).add_subplot(1, 1, 1)
plt.plot(t[790:],T[790:],'-', label='Temperature')
plt.xlabel(r"time t/s")
plt.ylabel(r"Temperature T / K")
plt.legend()
plt.xlim(750, 1400)
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

"""
# Plot R_P_2 over T, (R_P_2 = R_Probe_2/Ohm (Si))
fig = plt.figure(figsize=(8, 4), dpi=120).add_subplot(1, 1, 1)
plt.plot(T,R_P_2,'-', label='Resistance of Si')
plt.xlabel(r"Temperature T / K")
plt.ylabel(r"Resistance R / $\Omega$")
plt.legend()
#plt.xlim(0, 320)
#plt.ylim(0, 120)
#plt.yscale('log')
plt.title(r"Resistance of Si over Temperature")
plt.show()
"""


#...end_script1...#