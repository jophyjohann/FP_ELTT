from dataset_operations2 import DataSet_Operations2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import os

class run:
	def __init__(self):
		self.dat = DataSet_Operations2()
		self.dat.import_dataset_measurements()

		self.export_folder = "export/script" + __name__[-1] + "/"
		self.export_extension = ".pdf"


	def main(self):
		dataSet = self.dat.dataSet
		
		def maximize():
			'''maximizes the matplotlib plot window'''
			mng = plt.get_current_fig_manager()
			mng.resize(*mng.window.maxsize())
			#mng.full_screen_toggle()
			#os.system('xdotool key alt+F10')
		
		def logistic(x, a, b, c, d):
			return a / np.sqrt(1 + np.exp(b * (x + c))) + d

		def B_func(T, Tc, xi0):
			phi0 = 2.067833848e-15
			return (1 - (T / Tc)) * phi0 / (2 * np.pi * xi0**2)
		
		def offset(x, b):
			x = [b for i in x]
			return x
		
			
		print(80*"_"+"\n\nPlotting: RestWiderstand Niob")
		
		### Plot für RestWiderstand
		data = dataSet[0]
		
		plot_range = [1650, None]
		fit_range = [46, None]
		fit_plot_range = [46, None]
		
		# set plot characteristics of x and y ticks and grid
		major_x_ticks = 2
		minor_x_ticks = 1
		major_y_ticks = 0.002
		minor_y_ticks = 0.0004
		major_alpha = 0.7
		minor_alpha = 0.2
		
		T_data = data['T'][plot_range[0]:plot_range[1]]
		t_data = data['t'][plot_range[0]:plot_range[1]]
		R_data = data['R_P_1'][plot_range[0]:plot_range[1]]
			
		fit_parameters = [["b"],
										  [ 0.1],		 # max bounds
										  [0.015],		 # start values
										  [0]]		 # min bounds
		
		popt, pcov = curve_fit(offset, T_data[fit_range[0]:fit_range[1]], R_data[fit_range[0]:fit_range[1]], fit_parameters[2])#, bounds=(fit_parameters[3],fit_parameters[1]))  
			
		fig = plt.figure(figsize=(8, 4), dpi=120).add_subplot(1, 1, 1)
		#plt.plot(t_data, T_data, '.')
		plt.plot(T_data, R_data, '.')
		plt.plot(T_data[fit_plot_range[0]:fit_plot_range[1]], offset(T_data[fit_plot_range[0]:fit_plot_range[1]], *popt), '--', label = "Restwiderstand $R_R$=({:.4}$\pm${:.4})$\Omega$".format(popt[0],np.sqrt(np.diag(pcov))[0]))
		plt.xlabel("T / K")
		plt.ylabel(r"R / $\Omega$")
		fig.set_xticks(np.arange(plt.xlim()[0],plt.xlim()[1],major_x_ticks))
		fig.set_xticks(np.arange(plt.xlim()[0],plt.xlim()[1],minor_x_ticks),minor=True)
		fig.set_yticks(np.arange(plt.ylim()[0],plt.ylim()[1],major_y_ticks))
		fig.set_yticks(np.arange(plt.ylim()[0],plt.ylim()[1],minor_y_ticks),minor=True)
		fig.grid(which='minor', alpha=minor_alpha)
		fig.grid(which='major', alpha=major_alpha)
		plt.title("RestWiderstand")
		plt.legend()
		maximize()
		plt.show()


			
		### Plot für Phasendiagramm

		data = dataSet[1]
			
		plot_lims = [[705,950],
								[1100,1450],
								[1450,1850],
								[1850,2700],
								[2650,3150],
								[3150,3600],
								[3550,4350]]
		
		# set plot characteristics of x and y ticks and grid
		major_x_ticks = 500
		minor_x_ticks = 100
		major_y_ticks = 0.005
		minor_y_ticks = 0.001
		major_alpha = 0.7
		minor_alpha = 0.2
			
		T_data = data['T']#[plot_lims[0][0]:plot_lims[0][1]]
		t_data = data['t']#[plot_lims[0][0]:plot_lims[0][1]]
		R_data = data['R_P_1']#[plot_lims[0][0]:plot_lims[0][1]]
		
		fig, ax1 = plt.subplots()
		ax1.plot(t_data, R_data, '-')
		ax1.set_ylabel(r"R / $\Omega$",color="tab:blue")
		ax1.set_xlabel("t / s")
		ax1.tick_params(axis='y', labelcolor="tab:blue")
		
		ax2 = ax1.twinx()
		ax2.plot(t_data, T_data, 'k-')
		ax2.set_ylabel("T / K")
		
		#plt.plot(t_data, R_data, '-')
		#plt.xlim(0,None)
		#plt.ylim(0,None)
		#fig.set_xticks(np.arange(plt.xlim()[0],plt.xlim()[1],major_x_ticks))
		#fig.set_xticks(np.arange(plt.xlim()[0],plt.xlim()[1],minor_x_ticks),minor=True)
		#fig.set_yticks(np.arange(plt.ylim()[0],plt.ylim()[1],major_y_ticks))
		#fig.set_yticks(np.arange(plt.ylim()[0],plt.ylim()[1],minor_y_ticks),minor=True)
		#fig.grid(which='minor', alpha=minor_alpha)
		#fig.grid(which='major', alpha=major_alpha)
		maximize()
		fig.tight_layout()
		plt.show()


		
		print(80*"_"+"\n\nPlotting: Tc Plot für I = 0A")

		# fitting the function
		plot_range = [None, None]
		fit_range = [None, None]

		mess_no = 0
		t_data = data['t'][plot_lims[mess_no][0]:plot_lims[mess_no][1]]
		T_data = data['T'][plot_lims[mess_no][0]:plot_lims[mess_no][1]]
		R_data = data['R_P_1'][plot_lims[mess_no][0]:plot_lims[mess_no][1]]

		T_data = T_data[plot_range[0]:plot_range[1]]
		R_data = R_data[plot_range[0]:plot_range[1]]

		measurements = [T_data, R_data]
		
		#measurements = np.sort(measurements)
		T_data = measurements[0]
		R_data = measurements[1]
		
		fit_parameters = [["a","b",  "c","d"],
										  [ 0,  20, -6.9, 0.0655],		 # max bounds
										  [-0.01,  10, -9, 0.015],		 # start values
										  [-0.09, 0.1, -8.9,  0.06]]		 # min bounds
		
		popt, pcov = curve_fit(logistic, T_data[fit_range[0]:fit_range[1]], R_data[fit_range[0]:fit_range[1]], fit_parameters[2])#, bounds=(fit_parameters[3],fit_parameters[1]))  
			
		popt0= popt.copy()
		pcov0 = pcov.copy()
			
		print("\nLogistic Fkt. y = a/(1+exp(b*x +c) +d ")
		print("a = {:.4g} +\- {:.4g}\nb= {:.4g} +\- {:.4g}\nc= {:.4g} +\- {:.4g}\nd= {:.4g} +\- {:.4g}".format(popt[0], np.sqrt(np.diag(pcov))[0], popt[1], np.sqrt(np.diag(pcov))[1], popt[2], np.sqrt(np.diag(pcov))[2], popt[3], np.sqrt(np.diag(pcov))[3]))
		
		fig = plt.figure(figsize=(8, 4), dpi=120).add_subplot(1, 1, 1)
		#plt.plot(t_data, R_data, '.')
		plt.plot(T_data, R_data, '.')
		T_data = np.linspace(T_data[0],T_data[-1],1000)
		plt.plot(T_data[fit_range[0]:fit_range[1]], logistic(T_data[fit_range[0]:fit_range[1]], *popt), '-', label = r"Logistic func. fit")
		plt.xlabel("T / K")
		plt.ylabel(r"R / $\Omega$")
		plt.title("Tc Plot für I = 0A")
		maximize()
		#plt.show()


			
		print(80*"_"+"\n\nPlotting: Tc Plot für I = 1A")

		# fitting the function
		plot_range = [None, None]
		fit_range = [None, None]

		mess_no = 1
		t_data = data['t'][plot_lims[mess_no][0]:plot_lims[mess_no][1]]
		T_data = data['T'][plot_lims[mess_no][0]:plot_lims[mess_no][1]]
		R_data = data['R_P_1'][plot_lims[mess_no][0]:plot_lims[mess_no][1]]

		T_data = T_data[plot_range[0]:plot_range[1]]
		R_data = R_data[plot_range[0]:plot_range[1]]

		measurements = [T_data, R_data]
		
		#measurements = np.sort(measurements)
		T_data = measurements[0]
		R_data = measurements[1]
		
		fit_parameters = [["a","b",  "c","d"],
										  [ 0,  20, -6.9, 0.0655],		 # max bounds
										  [-0.01,  10, -9, 0.015],		 # start values
										  [-0.09, 0.1, -8.9,  0.06]]		 # min bounds
		
		popt, pcov = curve_fit(logistic, T_data[fit_range[0]:fit_range[1]], R_data[fit_range[0]:fit_range[1]], fit_parameters[2])#, bounds=(fit_parameters[3],fit_parameters[1]))  
			
		popt1= popt.copy()
		pcov1 = pcov.copy()
			
		print("\nLogistic Fkt. y = a/(1+exp(b*x +c) +d ")
		print("a = {:.4g} +\- {:.4g}\nb= {:.4g} +\- {:.4g}\nc= {:.4g} +\- {:.4g}\nd= {:.4g} +\- {:.4g}".format(popt[0], np.sqrt(np.diag(pcov))[0], popt[1], np.sqrt(np.diag(pcov))[1], popt[2], np.sqrt(np.diag(pcov))[2], popt[3], np.sqrt(np.diag(pcov))[3]))
		
		fig = plt.figure(figsize=(8, 4), dpi=120).add_subplot(1, 1, 1)
		#plt.plot(t_data, T_data, '.')
		plt.plot(T_data, R_data, '.')
		T_data = np.linspace(T_data[0],T_data[-1],1000)
		plt.plot(T_data[fit_range[0]:fit_range[1]], logistic(T_data[fit_range[0]:fit_range[1]], *popt), '-', label = r"Logistic func. fit")
		plt.xlabel("T / K")
		plt.ylabel(r"R / $\Omega$")
		plt.title("Tc Plot für I = 1A")
		maximize()
		#plt.show()

		
			
		print(80*"_"+"\n\nPlotting: Tc Plot für I = 2A")

		# fitting the function
		plot_range = [None, None]
		fit_range = [None, None]

		mess_no = 2
		t_data = data['t'][plot_lims[mess_no][0]:plot_lims[mess_no][1]]
		T_data = data['T'][plot_lims[mess_no][0]:plot_lims[mess_no][1]]
		R_data = data['R_P_1'][plot_lims[mess_no][0]:plot_lims[mess_no][1]]

		T_data = T_data[plot_range[0]:plot_range[1]]
		R_data = R_data[plot_range[0]:plot_range[1]]

		measurements = [T_data, R_data]
		
		#measurements = np.sort(measurements)
		T_data = measurements[0]
		R_data = measurements[1]
		
		fit_parameters = [["a","b",  "c","d"],
										  [ 0,  20, -6.9, 0.0655],		 # max bounds
										  [-0.01,  10, -9, 0.015],		 # start values
										  [-0.09, 0.1, -8.9,  0.06]]		 # min bounds
		
		popt, pcov = curve_fit(logistic, T_data[fit_range[0]:fit_range[1]], R_data[fit_range[0]:fit_range[1]], fit_parameters[2])#, bounds=(fit_parameters[3],fit_parameters[1]))  
			
		popt2= popt.copy()
		pcov2 = pcov.copy()
			
		print("\nLogistic Fkt. y = a/(1+exp(b*x +c) +d ")
		print("a = {:.4g} +\- {:.4g}\nb= {:.4g} +\- {:.4g}\nc= {:.4g} +\- {:.4g}\nd= {:.4g} +\- {:.4g}".format(popt[0], np.sqrt(np.diag(pcov))[0], popt[1], np.sqrt(np.diag(pcov))[1], popt[2], np.sqrt(np.diag(pcov))[2], popt[3], np.sqrt(np.diag(pcov))[3]))
		
		fig = plt.figure(figsize=(8, 4), dpi=120).add_subplot(1, 1, 1)
		#plt.plot(t_data, T_data, '.')
		plt.plot(T_data, R_data, '.')
		T_data = np.linspace(T_data[0],T_data[-1],1000)
		plt.plot(T_data[fit_range[0]:fit_range[1]], logistic(T_data[fit_range[0]:fit_range[1]], *popt), '-', label = r"Logistic func. fit")
		plt.xlabel("T / K")
		plt.ylabel(r"R / $\Omega$")
		plt.title("Tc Plot für I = 2A")
		maximize()
		#plt.show()

		
			
		print(80*"_"+"\n\nPlotting: Tc Plot für I = 3A")

		# fitting the function
		plot_range = [None, None]
		fit_range = [None, None]

		mess_no = 3
		t_data = data['t'][plot_lims[mess_no][0]:plot_lims[mess_no][1]]
		T_data = data['T'][plot_lims[mess_no][0]:plot_lims[mess_no][1]]
		R_data = data['R_P_1'][plot_lims[mess_no][0]:plot_lims[mess_no][1]]

		T_data = T_data[plot_range[0]:plot_range[1]]
		R_data = R_data[plot_range[0]:plot_range[1]]

		measurements = [T_data, R_data]
		
		#measurements = np.sort(measurements)
		T_data = measurements[0]
		R_data = measurements[1]
		
		fit_parameters = [["a","b",  "c","d"],
										  [ 0,  20, -6.9, 0.0655],		 # max bounds
										  [-0.01,  10, -9, 0.015],		 # start values
										  [-0.09, 0.1, -8.9,  0.06]]		 # min bounds
		
		popt, pcov = curve_fit(logistic, T_data[fit_range[0]:fit_range[1]], R_data[fit_range[0]:fit_range[1]], fit_parameters[2])#, bounds=(fit_parameters[3],fit_parameters[1]))  
			
		popt3= popt.copy()
		pcov3 = pcov.copy()
			
		print("\nLogistic Fkt. y = a/(1+exp(b*x +c) +d ")
		print("a = {:.4g} +\- {:.4g}\nb= {:.4g} +\- {:.4g}\nc= {:.4g} +\- {:.4g}\nd= {:.4g} +\- {:.4g}".format(popt[0], np.sqrt(np.diag(pcov))[0], popt[1], np.sqrt(np.diag(pcov))[1], popt[2], np.sqrt(np.diag(pcov))[2], popt[3], np.sqrt(np.diag(pcov))[3]))
		
		fig = plt.figure(figsize=(8, 4), dpi=120).add_subplot(1, 1, 1)
		#plt.plot(t_data, R_data, '.')
		plt.plot(T_data, R_data, '.')
		T_data = np.linspace(T_data[0],T_data[-1],1000)
		plt.plot(T_data[fit_range[0]:fit_range[1]], logistic(T_data[fit_range[0]:fit_range[1]], *popt), '-', label = r"Logistic func. fit")
		plt.xlabel("T / K")
		plt.ylabel(r"R / $\Omega$")
		plt.title("Tc Plot für I = 3A")
		maximize()
		#plt.show()

		
			
		print(80*"_"+"\n\nPlotting: Tc Plot für I = 4A")

		# fitting the function
		plot_range = [None, None]
		fit_range = [None, None]

		mess_no = 4
		t_data = data['t'][plot_lims[mess_no][0]:plot_lims[mess_no][1]]
		T_data = data['T'][plot_lims[mess_no][0]:plot_lims[mess_no][1]]
		R_data = data['R_P_1'][plot_lims[mess_no][0]:plot_lims[mess_no][1]]

		T_data = T_data[plot_range[0]:plot_range[1]]
		R_data = R_data[plot_range[0]:plot_range[1]]

		measurements = [T_data, R_data]
		
		#measurements = np.sort(measurements)
		T_data = measurements[0]
		R_data = measurements[1]
		
		fit_parameters = [["a","b",  "c","d"],
										  [ 0,  20, -6.9, 0.0655],		 # max bounds
										  [-0.01,  10, -7.5, 0.0155],		 # start values
										  [-0.09, 0.1, -8.9,  0.06]]		 # min bounds
		
		popt, pcov = curve_fit(logistic, T_data[fit_range[0]:fit_range[1]], R_data[fit_range[0]:fit_range[1]], fit_parameters[2])#, bounds=(fit_parameters[3],fit_parameters[1]))  
			
		popt4= popt.copy()
		pcov4 = pcov.copy()
			
		print("\nLogistic Fkt. y = a/(1+exp(b*x +c) +d ")
		print("a = {:.4g} +\- {:.4g}\nb= {:.4g} +\- {:.4g}\nc= {:.4g} +\- {:.4g}\nd= {:.4g} +\- {:.4g}".format(popt[0], np.sqrt(np.diag(pcov))[0], popt[1], np.sqrt(np.diag(pcov))[1], popt[2], np.sqrt(np.diag(pcov))[2], popt[3], np.sqrt(np.diag(pcov))[3]))
		
		fig = plt.figure(figsize=(8, 4), dpi=120).add_subplot(1, 1, 1)
		#plt.plot(t_data, T_data, '.')
		plt.plot(T_data, R_data, '.')
		T_data = np.linspace(T_data[0],T_data[-1],1000)
		plt.plot(T_data[fit_range[0]:fit_range[1]], logistic(T_data[fit_range[0]:fit_range[1]], *popt), '-', label = r"Logistic func. fit")
		plt.xlabel("T / K")
		plt.ylabel(r"R / $\Omega$")
		plt.title("Tc Plot für I = 4A")
		maximize()
		#plt.show()

		
			
		print(80*"_"+"\n\nPlotting: Tc Plot für I = 5A")

		# fitting the function
		plot_range = [None, None]
		fit_range = [None, None]

		mess_no = 5
		t_data = data['t'][plot_lims[mess_no][0]:plot_lims[mess_no][1]]
		T_data = data['T'][plot_lims[mess_no][0]:plot_lims[mess_no][1]]
		R_data = data['R_P_1'][plot_lims[mess_no][0]:plot_lims[mess_no][1]]

		T_data = T_data[plot_range[0]:plot_range[1]]
		R_data = R_data[plot_range[0]:plot_range[1]]

		measurements = [T_data, R_data]
		
		#measurements = np.sort(measurements)
		T_data = measurements[0]
		R_data = measurements[1]
		
		fit_parameters = [["a","b",  "c","d"],
										  [ 0,  20, -6.9, 0.0655],		 # max bounds
										  [-0.01,  10, -7.5, 0.0155],		 # start values
										  [-0.09, 0.1, -8.9,  0.06]]		 # min bounds
		
		popt, pcov = curve_fit(logistic, T_data[fit_range[0]:fit_range[1]], R_data[fit_range[0]:fit_range[1]], fit_parameters[2])#, bounds=(fit_parameters[3],fit_parameters[1]))  
			
		popt5= popt.copy()
		pcov5 = pcov.copy()
			
		print("\nLogistic Fkt. y = a/(1+exp(b*x +c) +d ")
		print("a = {:.4g} +\- {:.4g}\nb= {:.4g} +\- {:.4g}\nc= {:.4g} +\- {:.4g}\nd= {:.4g} +\- {:.4g}".format(popt[0], np.sqrt(np.diag(pcov))[0], popt[1], np.sqrt(np.diag(pcov))[1], popt[2], np.sqrt(np.diag(pcov))[2], popt[3], np.sqrt(np.diag(pcov))[3]))
		
		fig = plt.figure(figsize=(8, 4), dpi=120).add_subplot(1, 1, 1)
		#plt.plot(t_data, R_data, '.')
		plt.plot(T_data, R_data, '.')
		T_data = np.linspace(T_data[0],T_data[-1],1000)
		plt.plot(T_data[fit_range[0]:fit_range[1]], logistic(T_data[fit_range[0]:fit_range[1]], *popt), '-', label = r"Logistic func. fit")
		plt.xlabel("T / K")
		plt.ylabel(r"R / $\Omega$")
		plt.title("Tc Plot für I = 5A")
		maximize()
		#plt.show()

		
			
		print(80*"_"+"\n\nPlotting: Tc Plot für I = 6A")

		# fitting the function
		plot_range = [None, None]
		fit_range = [None, None]

		mess_no = 6
		t_data = data['t'][plot_lims[mess_no][0]:plot_lims[mess_no][1]]
		T_data = data['T'][plot_lims[mess_no][0]:plot_lims[mess_no][1]]
		R_data = data['R_P_1'][plot_lims[mess_no][0]:plot_lims[mess_no][1]]

		T_data = T_data[plot_range[0]:plot_range[1]]
		R_data = R_data[plot_range[0]:plot_range[1]]

		measurements = [T_data, R_data]
		
		#measurements = np.sort(measurements)
		T_data = measurements[0]
		R_data = measurements[1]
		
		fit_parameters = [["a","b",  "c","d"],
										  [ 0,  20, -6.9, 0.0655],		 # max bounds
										  [-0.01,  10, -7.5, 0.0155],		 # start values
										  [-0.09, 0.1, -8.9,  0.06]]		 # min bounds
		
		popt, pcov = curve_fit(logistic, T_data[fit_range[0]:fit_range[1]], R_data[fit_range[0]:fit_range[1]], fit_parameters[2])#, bounds=(fit_parameters[3],fit_parameters[1]))  
			
		popt6= popt.copy()
		pcov6 = pcov.copy()
			
		print("\nLogistic Fkt. y = a/(1+exp(b*x +c) +d ")
		print("a = {:.4g} +\- {:.4g}\nb= {:.4g} +\- {:.4g}\nc= {:.4g} +\- {:.4g}\nd= {:.4g} +\- {:.4g}".format(popt[0], np.sqrt(np.diag(pcov))[0], popt[1], np.sqrt(np.diag(pcov))[1], popt[2], np.sqrt(np.diag(pcov))[2], popt[3], np.sqrt(np.diag(pcov))[3]))
		
		fig = plt.figure(figsize=(8, 4), dpi=120).add_subplot(1, 1, 1)
		#plt.plot(t_data, T_data, '.')
		plt.plot(T_data, R_data, '.')
		T_data = np.linspace(T_data[0],T_data[-1],1000)
		plt.plot(T_data[fit_range[0]:fit_range[1]], logistic(T_data[fit_range[0]:fit_range[1]], *popt), '-', label = r"Logistic func. fit")
		plt.xlabel("T / K")
		plt.ylabel(r"R / $\Omega$")
		plt.title("Tc Plot für I = 6A")
		maximize()
		plt.show()


			
		print(80*"_"+"\n\nPlotting: Phasendiagramm Niob")
		
		# fitting the function
		fit_range = [None, None]

		T_data = [popt0[2],popt1[2],popt2[2],popt3[2],popt4[2],popt5[2],popt6[2]]
		T_data = [-i for i in T_data]
		I_data = [2, 3, 4, 5, 6, 7, 8]
		B_data = [i * 194/6 for i in I_data]	# 6A entspricht 194mT
		
		fit_parameters = [["Tc","Xi0"],
										  [8.9,  0.55e-9],		 # max bounds
										  [8.8,  0.50e-9],		 # start values
										  [8.7,  0.45e-9]]		 # min bounds
		
		popt, pcov = curve_fit(B_func, T_data[fit_range[0]:fit_range[1]][:3], B_data[fit_range[0]:fit_range[1]][:3], fit_parameters[2])#, bounds=(fit_parameters[3],fit_parameters[1]))  

		l = 1e9*popt[1] / 39
		print("Somit ergibt sich für die mittlere freihe Weglänge l={:.4}nm".format(l))
		
		fig = plt.figure(figsize=(8, 4), dpi=120).add_subplot(1, 1, 1)
		plt.plot(T_data, B_data, '.')
		T_data = np.linspace(T_data[0],T_data[-1],1000)
		plt.plot(T_data[fit_range[0]:fit_range[1]], B_func(T_data[fit_range[0]:fit_range[1]], *popt), '-', label = r"Lin. Fit mit $\xi^0$=({:.3}$\pm${:.3})nm".format(1e9*popt[1],1e9*np.sqrt(np.diag(pcov))[1])+"\n"+r"und $T_C$=({:.3}$\pm${:.3})K".format(popt[0],np.sqrt(np.diag(pcov))[0]))
		plt.xlabel(r"$T_C$ / K")
		plt.ylabel(r"$B_C$ / mT")
		plt.title("Phasendiagramm Niob")
		plt.legend()
		maximize()
		plt.show()