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

def logistic(x, a, b, c, d):
    return a / np.sqrt(1 + np.exp(b * (x + c))) + d

def func2(x, A, B, c):
    np.log(A*(1/x + c)**(-3/2)) - B/x

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
K_2=lin(300, popt[0], popt[1])
RRR=K_2/K_1
#print(popt)
print("Fit eq: y= m*x+n")
print("m= {:.4g} +/- {:.4g}, n= {:.4g} +/- {:.4g}".format(opt_fit_parameters2[0], np.sqrt(np.diag(pcov2))[0], opt_fit_parameters2[1], np.sqrt(np.diag(pcov2))[1]))
print("R(300K)= {:.4g}".format(K_2))
print("RRR=R(300K)/R(4.2K)= {:.4g}".format(RRR))

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

# R(T) = R_rest + R_T(T)
# R_rest = R(4,2K) = K_1
# R(T) = R_P_1
# daraus folgt R_T(T) = R_P_1-K_1
R_T = R_P_1 - K_1 # = 1.17R_T(Theta_D)*T/Theta_D -0.17R_T(Theta_D)
# Daraus folgt Anstieg m = 1,17 R_T(Theta_D)/Theta_D, und n = -0.17R_T(Theta_D)
# m und n oben aus linearem Fit bestimmt. (wobei von n der Wert von R_rest = K_1 abgezogen werden muss)
# daraus folgt Theta_D = -1.17*n/(0.17*m)
n = opt_fit_parameters2[1]
m = opt_fit_parameters2[0]
Theta_D_Cu = -1.17*(n-K_1)/(0.17*opt_fit_parameters2[0])

# Gausschefehlerfortpflanzung um Delta Theta_D zu bestimmen aus fit unsicherheiten von m und n
Delta_m = np.sqrt(np.diag(pcov2))[0]
Delta_n = np.sqrt(np.diag(pcov2))[1]
Delta_Theta_D = np.sqrt((Theta_D_Cu/n*Delta_n)**2 + (Theta_D_Cu/m*Delta_m)**2 )

print("Debye Temp von Cu Theta_D/K = {:.4g}+\- {:.4g}".format(Theta_D_Cu, Delta_Theta_D))

# Reduced data, ie y= R(T)/R(Theta_D), und x = T/Theta_D
# Resistivity is rho= (A/l)*R, Calling constant C = (A/l) = pi*r^2/l
C = np.pi*(0.08*10**(-3))/1  # Kürtzt sich einfach bei dem Plot 
R_Theta_D = lin(Theta_D_Cu, m, n)
rho = [T/Theta_D_Cu, R_P_1/R_Theta_D]

# Literature Values according to Instructions
Na_T = [0.388, 0.282]    # Reduced Temp
Na_R = [0.267, 0.152]    # Reduced Res
Au_T = [0.328, 0.120, 0.105]
Au_R = [0.207, 0.017, 0.009]
Cu_T = [0.266, 0.247, 0.082, 0.062]
Cu_R = [0.132, 0.113, 0.003, 0.002]
Al_T = [0.230, 0.218, 0.208, 0.202]
Al_R = [0.094, 0.079, 0.070, 0.061]
Ni_T = [0.172, 0.163]
Ni_R = [0.042, 0.038]

# Plot Reduced data, ie y= R(T)/R(Theta_D), und x = T/Theta_D
fig = plt.figure(figsize=(8, 4), dpi=120).add_subplot(1, 1, 1)
# Add literature Data...
plt.plot(rho[0], rho[1],'-', label='Reduced Resistivity')
plt.plot(Na_T, Na_R,'.', label='Na')
plt.plot(Au_T, Au_R,'.', label='Au')
plt.plot(Cu_T, Cu_R,'.', label='Cu')
plt.plot(Al_T, Al_R,'.', label='Al')
plt.plot(Ni_T, Ni_R,'.', label='Ni')

plt.xlabel(r"Reduced Temperature T/$\Theta_D$")
plt.ylabel(r"Resistance R(T) / R($\Theta_D$)")
plt.legend()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title(r"Reduced Resistivit of Cu over reduced Temperature")
plt.show()



# Plot R_T over T, (R_T = R_Thermometer/Ohm)
fig = plt.figure(figsize=(8, 4), dpi=120).add_subplot(1, 1, 1)
plt.plot(T,R_T,'-', label='Resistance of Thermometer 1')
plt.xlabel(r"Temperature T / K")
plt.ylabel(r"Resistance R / $\Omega$")
plt.legend()
#plt.xlim(0, 320)
#plt.ylim(0, 120)
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

# Plot ln(sigma) over 1/T, (R_P_2 = R_Probe_2/Ohm (Si))
sigma = 6.5*10**(-3)/(R_P_2*5.4*10**(-6))
fig = plt.figure(figsize=(8, 4), dpi=120).add_subplot(1, 1, 1)
plt.plot(1/T, np.log(sigma),'-', label='Conductance of Si')
plt.xlabel(r"1/Temperature 1/T / 1/K")
plt.ylabel(r"Conductivity ln($\sigma$)/ln(($\Omega$*m)^-1)")
plt.legend()
plt.xlim(0, 0.04)
#plt.ylim(0, 120)
#plt.yscale('log')
plt.title(r"Conductivity of Si over Temperature")
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
#plt.ylim(0, 0.120)
#plt.yscale('log')
plt.title(r"Resistance of Si over Temperature")
plt.show()

# Plot ln(sigma) over 1/T, (R_P_2 = R_Probe_2/Ohm (Si))
print(min(R_P_2))


sigma = 6.5*10**(-3)/(R_P_2*5.4*10**(-6))
fig = plt.figure(figsize=(8, 4), dpi=120).add_subplot(1, 1, 1)
plt.plot(T,sigma,'.', label='Conductance of Si')
plt.xlabel(r"Temperature")
plt.ylabel(r"simga")
plt.legend()
#plt.xlim(0, 0.04)
#plt.ylim(0, 120)
#plt.yscale('log')
plt.title(r"Resistance of Si over Temperature")
plt.show()

fig = plt.figure(figsize=(8, 4), dpi=120).add_subplot(1, 1, 1)
plt.plot(1/T,np.log(sigma),'.', label='Conductance of Si')
plt.xlabel(r"1/Temperature 1/T / 1/K")
plt.ylabel(r"Conductance ln($\sigma$)/ln(($\Omega$*m)^-1)")
plt.legend()
plt.xlim(0, 0.04)
#plt.ylim(0, 120)
#plt.yscale('log')
plt.title(r"Resistance of Si over Temperature")
plt.show()


# Load Nb-H-Field data
# t=time/s, T = Temp/Kelvin, R_P_1 = R_Probe_1/Ohm (Nb), R_T = R_Thermometer/Ohm, R_P_2 = R_Probe_2/Ohm (Si)
t, T, R_P_1, R_T, R_P_2 = np.loadtxt("Heinzelmann_Vincent_Nb_H-Feld.dat",  unpack = True, skiprows = 8)

# Approximation of the B-Field out of Equation from instructions from page 20 (very bottom)
# B = 0.03233(T/A)*I, comes from Biot-Savart at x=0
I_1 = 8.0 # A
I_2 = 4.0 # A
# Daraus folgt
B_1 =  0.03233*I_1
B_2 =  0.03233*I_2

# Plot T over t warming up and cooling down with B=0
fig = plt.figure(figsize=(8, 4), dpi=120).add_subplot(1, 1, 1)
plt.plot(t[:790],T[:790],'-', label='Temperature')
plt.xlabel(r"time t/s")
plt.ylabel(r"Temperature T / K")
plt.legend()
plt.xlim(0, 800)
plt.ylim(0, 12)
plt.title(r"Temperature over time warming up and cooling down")
plt.show()

# Plot T over t warming up and cooling down with B_1 and B_2 respectivly
fig = plt.figure(figsize=(8, 4), dpi=120).add_subplot(1, 1, 1)
plt.plot(t[790:],T[790:],'-', label='Temperature')
plt.axvline(x=1200, color='red', linestyle='dotted', label='Change from B_1 to B_2')
plt.xlabel(r"time t/s")
plt.ylabel(r"Temperature T / K")
plt.legend()
plt.xlim(750, 1400)
plt.ylim(0, 12)
plt.title(r"Temperature over time with different B-Fileds aplied")
plt.show()


# fitting the function
fit_range1 = [390, 500]
fit_range2 = [700, 730]
plot_range = [0,790]

fit_parameters_Nb_1 = [["a","b",  "c","d"],
                  [ 0,  20, -9, 0.0655],     # max bounds
                  [-0.01,  10, -9.2, 0.0645],     # start values
                  [-0.09, 0.1, -9.8,  0.06]]     # min bounds

popt, pcov = curve_fit(logistic, T[fit_range1[0]:fit_range1[1]], R_P_1[fit_range1[0]:fit_range1[1]], fit_parameters_Nb_1[2], bounds=(fit_parameters_Nb_1[3],fit_parameters_Nb_1[1]))  
popt2, pcov2 = curve_fit(logistic, T[fit_range2[0]:fit_range2[1]], R_P_1[fit_range2[0]:fit_range2[1]], fit_parameters_Nb_1[2], bounds=(fit_parameters_Nb_1[3],fit_parameters_Nb_1[1]))  

opt_fit_parameters_Nb_1 = popt.copy()
pcov_Nb_1 = pcov.copy()

opt_fit_parameters_Nb_2 = popt2.copy()
pcov_Nb_2 = pcov2.copy()

print("Logistic Fkt1. y = a/(1+exp(b*x +c) +d ")
print("a = {:.4g} +\- {:.4g}, b= {:.4g} +\- {:.4g}, c= {:.4g} +\- {:.4g}, d= {:.4g} +\- {:.4g}".format(opt_fit_parameters_Nb_1[0], np.sqrt(np.diag(pcov_Nb_1))[0], opt_fit_parameters_Nb_1[1], np.sqrt(np.diag(pcov_Nb_1))[1], opt_fit_parameters_Nb_1[2], np.sqrt(np.diag(pcov_Nb_1))[2], opt_fit_parameters_Nb_1[3], np.sqrt(np.diag(pcov_Nb_1))[3]))
print("Logistic Fkt2. y = a/(1+exp(b*x +c) +d ")
print("a = {:.4g} +\- {:.4g}, b= {:.4g} +\- {:.4g}, c= {:.4g} +\- {:.4g}, d= {:.4g} +\- {:.4g}".format(opt_fit_parameters_Nb_2[0], np.sqrt(np.diag(pcov_Nb_2))[0], opt_fit_parameters_Nb_2[1], np.sqrt(np.diag(pcov_Nb_2))[1], opt_fit_parameters_Nb_2[2], np.sqrt(np.diag(pcov_Nb_2))[2], opt_fit_parameters_Nb_2[3], np.sqrt(np.diag(pcov_Nb_2))[3]))

T_1 = np.linspace(9, 10, fit_range1[1]-fit_range1[0])
logistic1 = logistic(T_1, *popt)     # Trying something
logistic2 = logistic(T_1, *popt2)    # Works


# Plot R_P_1 over T, (R_P_1 = R_Probe_1/Ohm)(Nb)
fig = plt.figure(figsize=(8, 4), dpi=120).add_subplot(1, 1, 1)

plt.plot(T[:625],R_P_1[:625],'.', label='Resistance of Nb warming up')
plt.plot(T[625:790],R_P_1[625:790],'.', label='Resistance of Nb cooling down')
plt.plot(T_1, logistic1, 'g--', label="Logist. Fkt. 1 Fit warming up")
plt.plot(T_1, logistic2, 'r--', label="Logist. Fkt. 2 Fit cooling down")
plt.xlabel(r"Temperature T / K")
plt.ylabel(r"Resistance R / $\Omega$")
plt.legend()
#plt.xlim(0, 12)
plt.ylim(0.05, 0.075)
plt.title(r"Resistance of Nb over Temperature")
plt.show()

# fitting the function
fit_range1 = [1040, 1090]
fit_range2 = [1250, 1280]
plot_range = [0,790]

fit_parameters_Nb_3 = [["a","b",  "c","d"],
                  [ 0,  20, -6.9, 0.0655],     # max bounds
                  [-0.01,  10, -7.9, 0.0645],     # start values
                  [-0.09, 0.1, -8.9,  0.06]]     # min bounds

popt, pcov = curve_fit(logistic, T[fit_range1[0]:fit_range1[1]], R_P_1[fit_range1[0]:fit_range1[1]], fit_parameters_Nb_3[2], bounds=(fit_parameters_Nb_3[3],fit_parameters_Nb_3[1]))  
popt4, pcov4 = curve_fit(logistic, T[fit_range2[0]:fit_range2[1]], R_P_1[fit_range2[0]:fit_range2[1]], fit_parameters_Nb_3[2], bounds=(fit_parameters_Nb_3[3],fit_parameters_Nb_3[1]))  

opt_fit_parameters_Nb_3 = popt.copy()
pcov_Nb_3 = pcov.copy()

opt_fit_parameters_Nb_4 = popt4.copy()
pcov_Nb_4 = pcov4.copy()

print("Logistic Fkt3. y = a/(1+exp(b*x +c) +d ")
print("a = {:.4g} +\- {:.4g}, b= {:.4g} +\- {:.4g}, c= {:.4g} +\- {:.4g}, d= {:.4g} +\- {:.4g}".format(opt_fit_parameters_Nb_3[0], np.sqrt(np.diag(pcov_Nb_3))[0], opt_fit_parameters_Nb_3[1], np.sqrt(np.diag(pcov_Nb_3))[1], opt_fit_parameters_Nb_3[2], np.sqrt(np.diag(pcov_Nb_3))[2], opt_fit_parameters_Nb_3[3], np.sqrt(np.diag(pcov_Nb_3))[3]))
print("Logistic Fkt4. y = a/(1+exp(b*x +c) +d ")
print("a = {:.4g} +\- {:.4g}, b= {:.4g} +\- {:.4g}, c= {:.4g} +\- {:.4g}, d= {:.4g} +\- {:.4g}".format(opt_fit_parameters_Nb_4[0], np.sqrt(np.diag(pcov_Nb_4))[0], opt_fit_parameters_Nb_4[1], np.sqrt(np.diag(pcov_Nb_4))[1], opt_fit_parameters_Nb_4[2], np.sqrt(np.diag(pcov_Nb_4))[2], opt_fit_parameters_Nb_4[3], np.sqrt(np.diag(pcov_Nb_4))[3]))


# Plot R_P_1 over T, (R_P_1 = R_Probe_1/Ohm)(Nb)
fig = plt.figure(figsize=(8, 4), dpi=120).add_subplot(1, 1, 1)
plt.plot(T[790:1200],R_P_1[790:1200],'.', label='Resistance of Nb warming up with B_1')
plt.plot(T[1200:],R_P_1[1200:],'.', label='Resistance of Nb cooling down with B_2')
plt.plot(T[fit_range1[0]:fit_range1[1]],logistic(T[fit_range1[0]:fit_range1[1]], *opt_fit_parameters_Nb_3),'--', label='Logistic Fkt. Fit 1 warming up with B_1')
plt.plot(T[fit_range2[0]:fit_range2[1]],logistic(T[fit_range2[0]:fit_range2[1]], *opt_fit_parameters_Nb_4),'--', label='Logistic Fkt. Fit 2 cooling down with B_2')

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