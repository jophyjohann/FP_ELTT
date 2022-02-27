from dataset_operations import DataSet_Operations
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import *
from scipy.interpolate import make_interp_spline

class run:
	def __init__(self):
		self.dat = DataSet_Operations()
		self.dat.import_dataset_measurements()

		self.export_folder = "export/script" + __name__[-1] + "/"
		self.export_extension = ".png"
		self.dpi = 400
		self.figsize0 = (4.5, 4.5)
		self.figsize = (6.5, 4.5)
		self.figsize2 = (7.5, 4.5)
		self.figsize3 = (8.5, 4.5)
		self.markersize = 7
		self.markersize2 = 15
		self.linewidth_fit = 2


	def main(self):
		dataSet = self.dat.dataSet
		
		def maximize():
			'''maximizes the matplotlib plot window'''
			mng = plt.get_current_fig_manager()
			mng.resize(*mng.window.maxsize())
			#mng.full_screen_toggle()
			#os.system('xdotool key alt+F10')
		
		def exp(x,C,a,b,d):
			return C*np.exp(a*(x+b))+d
		
		def lin(x,m,n):
			return m*x+n
		
		def func1(x,a,b,c):
			return a*(x+b)**3 + c
		
		def logistic(x, a, b, c, d):
			return a / np.sqrt(1 + np.exp(b * (x + c))) + d
		
		def func2(x, A, B, c, d, e):
			return -np.log(A*(1/(e*(x + c)))**(-3/2)) - B/(e*(x + c)) + d
		
		plt.rcParams['font.size'] = '15'
		#'''
		plt.rcParams['axes.edgecolor'] = 'w'
		plt.rcParams['axes.labelcolor'] = 'w'
		plt.rcParams['figure.edgecolor'] = 'w'
		plt.rcParams['legend.edgecolor'] = 'w'
		plt.rcParams['text.color'] = 'w'
		plt.rcParams['legend.facecolor'] = 'none'
		plt.rcParams['xtick.color'] = 'w'
		plt.rcParams['ytick.color'] = 'w'
		
		plt.rcParams['savefig.transparent'] = 'True'
		#'''
		
		
		data = dataSet[2]
		t, T, R_P_1, R_T, R_P_2 = data['t'],data['T'],data['R_P_1'],data['R_T'],data['R_P_2']
		
		# Plot T over t
		print(r"Temperature over time of the Warm up process")
		fig = plt.figure(figsize=self.figsize2, dpi=80).add_subplot(1, 1, 1)
		plt.plot(t,T,'.', label='Temperature', color = "deepskyblue", markersize=self.markersize)
		plt.xlabel(r"Zeit t / s")
		plt.ylabel(r"Temperatur T / K")
		#plt.legend()
		plt.xlim(0, 5500)
		plt.ylim(0, 320)
		maximize()
		plt.savefig(self.export_folder+"T(t)_warm_up"+self.export_extension, bbox_inches='tight', dpi=self.dpi)
		plt.show()
		
		
		# Plot R_P_1 over T, (R_P_1 = R_Probe_1/Ohm)(Cu)
		print(r"Resistance of Cu over Temperature")
		fig = plt.figure(figsize=self.figsize2, dpi=80).add_subplot(1, 1, 1)
		plt.plot(T,R_P_1,'.', label='Widerstand von Cu', color = "deepskyblue", markersize=self.markersize)
		plt.xlabel(r"Temperatur T / K")
		plt.ylabel(r"Cu Widerstand R / $\Omega$")
		#plt.legend()
		plt.xlim(0, 320)
		plt.ylim(0, 3)
		maximize()
		plt.savefig(self.export_folder+"R_Cu(T)"+self.export_extension, bbox_inches='tight', dpi=self.dpi)
		plt.show()
		
		'''
		# fitting the T^3 function
		plot_range = [0,1087]
		fit_range = [0,800]
		
		fit_parameters = [["a" ,"b", "c"],   
										  [ 1e-5,		 -0.01, 0.1],		  # max bounds
										  [2e-6,   -4.2, 0.07],		# start values
										  [1e-6, -10, 0]]		 # min bounds
		
		popt, pcov = curve_fit(func1, T[fit_range[0]:fit_range[1]], R_P_1[fit_range[0]:fit_range[1]], fit_parameters[2], bounds=(fit_parameters[3],fit_parameters[1]))
		
		opt_fit_parameters1 = popt.copy()
		pcov1 = pcov.copy()
		K_1=func1(4.2, popt[0], popt[1], popt[2])
		print("Fit eq: y= a*(x+b)^3")
		print("a= {:.4g} +/- {:.4g}, b= {:.4g} +/- {:.4g}, c= {:.4g} +/- {:.4g}".format(opt_fit_parameters1[0], np.sqrt(np.diag(pcov))[0], opt_fit_parameters1[1], np.sqrt(np.diag(pcov))[1], opt_fit_parameters1[2], np.sqrt(np.diag(pcov))[2]))
		print("R(4.2K)= {:.4g}".format(K_1))

		T = T[plot_range[0]:plot_range[1]]
		R_P_1 = R_P_1[plot_range[0]:plot_range[1]]
		# Plot R_P_1 over T, (R_P_1 = R_Probe_1/Ohm)(Cu) reduced range with T^3 fit
		print(r"Resistance of Cu over Temperature")
		fig = plt.figure(figsize=self.figsize, dpi=80).add_subplot(1, 1, 1)
		plt.plot(T, R_P_1, '.', label=r'$R_{Cu}$', color = "deepskyblue", markersize=self.markersize)
		T = np.linspace(T[0],T[-1],1000)
		plt.plot(T[fit_range[0]:fit_range[1]], func1(T[fit_range[0]:fit_range[1]], *popt), 'r--', label=r"Fit: $R = a\cdot(T+b)^3 + c$", linewidth=self.linewidth_fit)	
		plt.xlabel(r"Temperatur T / K")
		plt.ylabel(r"Widerstand $R_{Cu}$ / $\Omega$")
		plt.legend(markerscale=2)
		plt.xlim(0, None)
		#plt.ylim(0, 3)
		maximize()
		plt.savefig(self.export_folder+"R_Cu(T)_T3_Fit"+self.export_extension, bbox_inches='tight', dpi=self.dpi)
		plt.show()
		'''

		
		t, T, R_P_1, R_T, R_P_2 = data['t'],data['T'],data['R_P_1'],data['R_T'],data['R_P_2']
		
		# fitting the linear T function with T^3
		plot_range = [2647,3850]
			
		# fitting the T^3 function
		fit_range3 = [0,1500]
		
		fit_parameters = [["a" ,"b", "c"],   
										  [ 1e-5,		 -0.01, 0.1],		  # max bounds
										  [2e-6,   -4.2, 0.07],		# start values
										  [1e-6, -10, 0]]		 # min bounds
		
		popt, pcov = curve_fit(func1, T[fit_range3[0]:fit_range3[1]], R_P_1[fit_range3[0]:fit_range3[1]], fit_parameters[2], bounds=(fit_parameters[3],fit_parameters[1]))
		
		popt1 = popt.copy()
		pcov1 = pcov.copy()
		
		K_1=func1(4.2, popt[0], popt[1], popt[2])
		print("Fit eq: y= a*(x+b)^3")
		print("a= {:.4g} +/- {:.4g}, b= {:.4g} +/- {:.4g}, c= {:.4g} +/- {:.4g}".format(popt1[0], np.sqrt(np.diag(pcov))[0], popt1[1], np.sqrt(np.diag(pcov))[1], popt1[2], np.sqrt(np.diag(pcov1))[2]))
		print("R(4.2K)= {:.4g}".format(K_1))

		# fitting the linear T function
		fit_range = [2647,4200]
		
		fit_parameters = [["m"   ,  "n"],   
										  [   0.1,  0.0],		  # max bounds
										  [  0.01, -0.5],		# start values
										  [ 0.001,   -2]]		 # min bounds
			
		popt, pcov = curve_fit(lin, T[fit_range[0]:fit_range[1]], R_P_1[fit_range[0]:fit_range[1]], fit_parameters[2], bounds=(fit_parameters[3],fit_parameters[1]))
		
		popt2 = popt.copy()
		pcov2 = pcov.copy()
		K_2=lin(300, popt[0], popt[1])
		RRR=K_2/K_1
		print("Fit eq: y= m*x+n")
		print("m= {:.4g} +/- {:.4g}, n= {:.4g} +/- {:.4g}".format(popt2[0], np.sqrt(np.diag(pcov2))[0], popt2[1], np.sqrt(np.diag(pcov2))[1]))
		print("R(300K)= {:.4g}".format(K_2))
		print("RRR=R(300K)/R(4.2K)= {:.4g}".format(RRR))
			
		# Plot R_P_1 over T, (R_P_1 = R_Probe_1/Ohm)(Cu) reduced range with T^1 fit
		print(r"Resistance of Cu over Temperature")
		fig = plt.figure(figsize=self.figsize, dpi=80).add_subplot(1, 1, 1)
		plt.plot(T,R_P_1,'.',  label=r'$R_{Cu}$', color = "deepskyblue", markersize=self.markersize)
		plt.plot(T[fit_range3[0]:fit_range3[1]], func1(T[fit_range3[0]:fit_range3[1]], *popt1), 'r--', label=r"Fit: $R = a\cdot(T+b)^3 + c$"+"\n"+r"$\Rightarrow$ R(4.2K)={:.4}$\Omega$".format(K_1), linewidth=self.linewidth_fit)
		plt.plot(T[fit_range[0]:fit_range[1]], lin(T[fit_range[0]:fit_range[1]], *popt2), '--', color="tab:orange", label=r"Lin. Fit: $R = m\cdot T + n$"+"\n"+r"$\Rightarrow$ R(300K)={:.4}$\Omega$".format(K_2), linewidth=self.linewidth_fit)
		plt.xlabel(r"Temperatur T / K")
		plt.ylabel(r"Widerstand R / $\Omega$")
		plt.legend(markerscale=2)
		plt.xlim(0, None)
		plt.ylim(0, 3.5)
		maximize()
		plt.savefig(self.export_folder+"R_Cu(T)_T_Fit"+self.export_extension, bbox_inches='tight', dpi=self.dpi)
		plt.show()

		
		# R(T) = R_rest + R_T(T)
		# R_rest = R(4,2K) = K_1
		# R(T) = R_P_1
		# daraus folgt R_T(T) = R_P_1-K_1
		R_T = R_P_1 - K_1 # = 1.17R_T(Theta_D)*T/Theta_D -0.17R_T(Theta_D)
		# Daraus folgt Anstieg m = 1,17 R_T(Theta_D)/Theta_D, und n = -0.17R_T(Theta_D)
		# m und n oben aus linearem Fit bestimmt. (wobei von n der Wert von R_rest = K_1 abgezogen werden muss)
		# daraus folgt Theta_D = -1.17*n/(0.17*m)
		n = popt2[1]
		m = popt2[0]
		Theta_D_Cu = -1.17*(n-K_1)/(0.17*popt2[0])
		
		# Gausschefehlerfortpflanzung um Delta Theta_D zu bestimmen aus fit unsicherheiten von m und n
		Delta_m = np.sqrt(np.diag(pcov2))[0]
		Delta_n = np.sqrt(np.diag(pcov2))[1]
		Delta_Theta_D = np.sqrt((Theta_D_Cu/n*Delta_n)**2 + (Theta_D_Cu/m*Delta_m)**2 )
		
		print("Debye Temp von Cu Theta_D/K = {:.4g}+\- {:.4g}".format(Theta_D_Cu, Delta_Theta_D))
		
		# Reduced data, ie y= R(T)/R(Theta_D), und x = T/Theta_D
		# Resistivity is rho= (A/l)*R, Calling constant C = (A/l) = pi*r^2/l
		C = np.pi*(0.08*10**(-3))/1  # Kürtzt sich einfach bei dem Plot 
		R_Theta_D = lin(Theta_D_Cu, m, n)
		rho = [T/Theta_D_Cu, R_T/R_Theta_D]
		
		# Literature Values according to Instructions
		Na_T = [0.388, 0.282]		# Reduced Temp
		Na_R = [0.267, 0.152]		# Reduced Res
		Au_T = [0.328, 0.120, 0.105]
		Au_R = [0.207, 0.017, 0.009]
		Cu_T = [0.266, 0.247, 0.082, 0.062]
		Cu_R = [0.132, 0.113, 0.003, 0.002]
		Al_T = [0.230, 0.218, 0.208, 0.202]
		Al_R = [0.094, 0.079, 0.070, 0.061]
		Ni_T = [0.172, 0.163]
		Ni_R = [0.042, 0.038]
		
		# Plot Reduced data, ie y= R(T)/R(Theta_D), und x = T/Theta_D
		print(r"Reduced Resistivity of Cu over reduced Temperature")
		fig = plt.figure(figsize=self.figsize, dpi=80).add_subplot(1, 1, 1)
		plt.plot(rho[0], rho[1],'.', label=r'Cu-Probe', color = "deepskyblue", markersize=self.markersize)
		plt.plot(Na_T, Na_R,'.', color="tab:orange", label='Na', markersize=self.markersize2)
		plt.plot(Au_T, Au_R,'.', color="tab:green", label='Au', markersize=self.markersize2)
		plt.plot(Cu_T, Cu_R,'.', color="tab:red", label='Cu', markersize=self.markersize2)
		plt.plot(Al_T, Al_R,'.', color="tab:purple", label='Al', markersize=self.markersize2)
		plt.plot(Ni_T, Ni_R,'.', color="tab:brown", label='Ni', markersize=self.markersize2)
		
		plt.xlabel(r"Reduz. Temperatur T / $\Theta_D$")
		plt.ylabel(r"Reduz. Widerstand R / R($\Theta_D$)")
		plt.legend(markerscale=2)
		plt.xlim(0, 0.41)
		plt.ylim(0, 0.3)
		maximize()
		plt.savefig(self.export_folder+"R_Cu(T)_red"+self.export_extension, bbox_inches='tight', dpi=self.dpi)
		plt.show()

		
		
		# Plot R_P_2 over T, (R_P_2 = R_Probe_2/Ohm (Si))
		print(r"Resistance of Si over Temperature")
		fig = plt.figure(figsize=self.figsize0, dpi=80).add_subplot(1, 1, 1)
		plt.plot(T,R_P_2/1e6, '.', label='Widerstand Si', color = "deepskyblue", markersize=self.markersize)
		plt.xlabel(r"Temperatur T / K")
		plt.ylabel(r"Si Widerstand R / M$\Omega$")
		plt.xlim(25, 35)
		#plt.ylim(0, 120)
		#plt.yscale('log')
		maximize()
		plt.savefig(self.export_folder+"R_Si(T)"+self.export_extension, bbox_inches='tight', dpi=self.dpi)
		plt.show()


			
		# Plot R_P_2 over T, (R_P_2 = R_Probe_2/Ohm (Si)) 2
		print(r"Resistance of Si over Temperature 2")
		fig = plt.figure(figsize=self.figsize0, dpi=80).add_subplot(1, 1, 1)
		plt.plot(T,R_P_2, '.', label='Widerstand Si', color = "deepskyblue", markersize=self.markersize)
		plt.xlabel(r"Temperatur T / K")
		plt.ylabel(r"Si Widerstand R / $\Omega$")
		plt.xlim(185, 255)
		plt.ylim(0, 260)
		#plt.yscale('log')
		fig.yaxis.set_label_position("right")
		fig.yaxis.tick_right()
		maximize()
		plt.savefig(self.export_folder+"R_Si(T)_2"+self.export_extension, bbox_inches='tight', dpi=self.dpi)
		plt.show()


			
		data = dataSet[1]
		t, T, R_P_1, R_T, R_P_2 = data['t'],data['T'],data['R_P_1'],data['R_T'],data['R_P_2']
		
		# Plot T over t
		print(r"Temperature over time of the cool down process")
		fig = plt.figure(figsize=self.figsize2, dpi=80).add_subplot(1, 1, 1)
		plt.plot(t,T,'.', label='Temperature', color = "deepskyblue", markersize=self.markersize)
		plt.xlabel(r"Zeit t / s")
		plt.ylabel(r"Temperatur T / K")
		plt.xlim(0, None)
		plt.ylim(0, None)
		maximize()
		plt.savefig(self.export_folder+"T(t)_cool_down"+self.export_extension, bbox_inches='tight', dpi=self.dpi)
		plt.show()
		

			
		# Plot R_P_1 over T, (R_P_1 = R_Probe_1/Ohm)(Nb)
		print(r"Resistance of Nb over Temperature")
		fig = plt.figure(figsize=self.figsize2, dpi=80).add_subplot(1, 1, 1)
		plt.plot(T,R_P_1,'.', label='Resistance of Nb', color = "deepskyblue", markersize=self.markersize)
		plt.xlabel(r"Temperatur T / K")
		plt.ylabel(r"Nb Widerstand R / $\Omega$")
		plt.xlim(0, None)
		#plt.ylim(0, None)
		maximize()
		plt.savefig(self.export_folder+"R_Nb(T)"+self.export_extension, bbox_inches='tight', dpi=self.dpi)
		plt.show()


			
		# Plot ln(sigma) over 1/T, (R_P_2 = R_Probe_2/Ohm (Si))
		R_P_2+=np.abs(min(R_P_2))+100
		sigma = 6.5*10**(-3)/(R_P_2*5.4*10**(-6))
											
		x = 1/T
		y= np.log(sigma)
		
		# linear fit
		plot_range = [0,3630]
		fit_range2 = [3550, 3603]
		fit_plot_range2 = [3300, 3603]
		
		fit_parameters_Si_lin = [["m", "n"],
														 [-1  ,   1]]		 # start values
		
		popt, pcov = curve_fit(lin, x[fit_range2[0]:fit_range2[1]], y[fit_range2[0]:fit_range2[1]], fit_parameters_Si_lin[1])  
		popt_sigma_lin = popt.copy()
		pcov_sigma_lin = pcov.copy()
		
		
		print("ln(sigma) of Si over inverse Temperature")
		
		e=1.602176634e-19 
		kbt=(1.38064852e-23)/e # eV/K
		
		print("Parameter der lin Fits:\n")
		print("m = {:.4g} +/- {:.4g}, n = {:.4g} +/- {:.4g}".format(popt_sigma_lin[0], np.diag(pcov_sigma_lin)[0], popt_sigma_lin[1], np.diag(pcov_sigma_lin)[1]))
		print("E_Don = {:.4g} +/- {:.4g} eV".format(-2*popt_sigma_lin[0]*kbt, np.abs(-2*kbt*np.diag(pcov_sigma_lin)[0])))
		E_donor_lit = 0.045 # eV
		print("E_Don_lit - E_Don_1 = {:.4g}".format(np.abs(E_donor_lit + 2*popt_sigma_lin[0]*kbt)))

		
		fig = plt.figure(figsize=self.figsize2, dpi=80).add_subplot(1, 1, 1)
		plt.plot(x[plot_range[0]:plot_range[1]], y[plot_range[0]:plot_range[1]],'.', label='ln($\sigma$)', color = "deepskyblue", markersize=self.markersize)
		plt.plot(x[fit_plot_range2[0]:fit_plot_range2[1]], lin(x, *popt_sigma_lin)[fit_plot_range2[0]:fit_plot_range2[1]], 'r--', label=r"Lin. Fit: $R = m\cdot\frac{1}{T} + n$"+"\n"+r"m={:.4}$\pm${:.3}".format(popt_sigma_lin[0],np.sqrt(np.diag(pcov_sigma_lin))[0]), linewidth=self.linewidth_fit)
		plt.xlabel(r"reziproke Temperatur 1 / T")
		plt.ylabel(r"ln($\sigma$)")
		plt.legend(markerscale=2, loc="lower left")
		plt.xlim(0, None)
		#plt.ylim(0, 120)
		maximize()
		plt.savefig(self.export_folder+"LN_Sigma_Si(inv_T)"+self.export_extension, bbox_inches='tight', dpi=self.dpi)
		plt.show()