from dataset_operations import DataSet_Operations
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import os

class run:
	def __init__(self):
		self.dat = DataSet_Operations()
		self.dat.import_dataset_measurements()

		self.export_folder = "export/script" + __name__[-1] + "/"
		self.export_extension = ".png"
		self.dpi = 400
		self.figsize = (6.5, 4.5)
		self.markersize = 10


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

		def lin(x, a, S):
			#return (1 - (T / Tc)) * phi0 / (2 * np.pi * xi0**2)
			return S * x + a
		
		def offset(x, b):
			x = [b for i in x]
			return x

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
		
		print(80*"_"+"\n\nSupraleitung Auswertung2 eigene Messdaten")

		'''
		print(80*"_"+"\n\nPlotting: Tc Plot für B = 0 (0)")

		# fitting the function
		plot_range = [-440, -390]
		fit_range = [None, None]
		
		fit_range_os_l = [-16, None]
		fit_plot_range_os_l = [None, None]
		#fit_plot_range_os_l = fit_range_os_l
		
		fit_range_os_r = [None, -20]
		fit_plot_range_os_r = [None, None]
		#fit_plot_range_os_r = fit_range_os_r
			
		data = dataSet[1]
		t_data = data['t']
		T_data = data['T']
		R_data = data['R_P_1']

		t_data = t_data[plot_range[0]:plot_range[1]]
		T_data = T_data[plot_range[0]:plot_range[1]]
		R_data = R_data[plot_range[0]:plot_range[1]]

		popt_os_l, pcov_os_l = curve_fit(offset, T_data[fit_range_os_l[0]:fit_range_os_l[1]], R_data[fit_range_os_l[0]:fit_range_os_l[1]])
		popt_os_r, pcov_os_r = curve_fit(offset, T_data[fit_range_os_r[0]:fit_range_os_r[1]], R_data[fit_range_os_r[0]:fit_range_os_r[1]])

		R_p_10p = (popt_os_r[0] - popt_os_l[0]) * 0.1 + popt_os_l[0]
		search_points = R_data[fit_range_os_l[1]:fit_range_os_l[0]]
		R_10p = search_points[np.abs(search_points-R_p_10p).argmin()]
		Tc_10p = min(T_data[np.where(R_data==R_10p)])
		print("Tc @ 10% = ",Tc_10p)
		
		R_p_90p = (popt_os_r[0] - popt_os_l[0]) * 0.9 + popt_os_l[0]
		search_points = R_data[fit_range_os_r[1]:fit_range_os_r[0]]
		R_90p = search_points[np.abs(search_points-R_p_90p).argmin()]
		Tc_90p = min(T_data[np.where(R_data==R_90p)])
		print("Tc @ 90% = ",Tc_90p)
		
		fig = plt.figure(figsize=(8, 4), dpi=160).add_subplot(1, 1, 1)
		plt.plot(T_data, R_data, '.')
		T_data = np.linspace(T_data[0],T_data[-1],1000)
		plt.plot(T_data[fit_plot_range_os_l[0]:fit_plot_range_os_l[1]], offset(T_data[fit_plot_range_os_l[0]:fit_plot_range_os_l[1]], *popt_os_l), '--')
		plt.plot(T_data[fit_plot_range_os_r[0]:fit_plot_range_os_r[1]], offset(T_data[fit_plot_range_os_r[0]:fit_plot_range_os_r[1]], *popt_os_r), '--')
		plt.plot(Tc_10p, R_10p, 'o')
		plt.plot(Tc_90p, R_90p, 'o')
		plt.xlabel("T / K")
		plt.ylabel(r"R / $\Omega$")
		#plt.title("Tc Plot für I = 0A")
		maximize()
		plt.show()


			
		print(80*"_"+"\n\nPlotting: Tc Plot für B = 0 (2)")

		# fitting the function
		plot_range = [240, 600]
		fit_range = [None, None]
		
		fit_range_os_l = [None, 150]
		fit_plot_range_os_l = [None, None]
		#fit_plot_range_os_l = fit_range_os_l
		
		fit_range_os_r = [250, None]
		fit_plot_range_os_r = [None, None]
		#fit_plot_range_os_r = fit_range_os_r
			
		data = dataSet[2]
		t_data = data['t']
		T_data = data['T']
		R_data = data['R_P_1']

		t_data = t_data[plot_range[0]:plot_range[1]]
		T_data = T_data[plot_range[0]:plot_range[1]]
		R_data = R_data[plot_range[0]:plot_range[1]]

		popt_os_l, pcov_os_l = curve_fit(offset, T_data[fit_range_os_l[0]:fit_range_os_l[1]], R_data[fit_range_os_l[0]:fit_range_os_l[1]])
		popt_os_r, pcov_os_r = curve_fit(offset, T_data[fit_range_os_r[0]:fit_range_os_r[1]], R_data[fit_range_os_r[0]:fit_range_os_r[1]])

		R_p_10p = (popt_os_r[0] - popt_os_l[0]) * 0.1 + popt_os_l[0]
		search_points = R_data[fit_range_os_l[1]:fit_range_os_l[0]]
		R_10p = search_points[np.abs(search_points-R_p_10p).argmin()]
		Tc_10p = min(T_data[np.where(R_data==R_10p)])
		print("Tc @ 10% = ",Tc_10p)
			
		R_p_90p = (popt_os_r[0] - popt_os_l[0]) * 0.9 + popt_os_l[0]
		search_points = R_data[fit_range_os_r[1]:fit_range_os_r[0]]
		R_90p = search_points[np.abs(search_points-R_p_90p).argmin()]
		Tc_90p = min(T_data[np.where(R_data==R_90p)])
		print("Tc @ 90% = ",Tc_90p)
		
		fig = plt.figure(figsize=(8, 4), dpi=160).add_subplot(1, 1, 1)
		plt.plot(T_data, R_data, '.')
		T_data = np.linspace(T_data[0],T_data[-1],1000)
		plt.plot(T_data[fit_plot_range_os_l[0]:fit_plot_range_os_l[1]], offset(T_data[fit_plot_range_os_l[0]:fit_plot_range_os_l[1]], *popt_os_l), '--')
		plt.plot(T_data[fit_plot_range_os_r[0]:fit_plot_range_os_r[1]], offset(T_data[fit_plot_range_os_r[0]:fit_plot_range_os_r[1]], *popt_os_r), '--')
		plt.plot(Tc_10p, R_10p, 'o')
		plt.plot(Tc_90p, R_90p, 'o')
		plt.xlabel("T / K")
		plt.ylabel(r"R / $\Omega$")
		#plt.title("Tc Plot für I = 0A")
		maximize()
		plt.show()
		'''

			
		print(80*"_"+"\n\nPlotting: Tc Plot für B = 0")

		# fitting the function
		plot_range = [670, 770]
		fit_range = [None, None]
		
		fit_range_os_l = [-50, None]
		fit_plot_range_os_l = [None, None]
		#fit_plot_range_os_l = fit_range_os_l
		
		fit_range_os_r = [None, -65]
		fit_plot_range_os_r = [None, None]
		#fit_plot_range_os_r = fit_range_os_r
			
		data = dataSet[0]
		t_data = data['t']
		T_data = data['T']
		R_data = data['R_P_1']

		t_data = t_data[plot_range[0]:plot_range[1]]
		T_data = T_data[plot_range[0]:plot_range[1]]
		R_data = R_data[plot_range[0]:plot_range[1]]

		popt_os_l, pcov_os_l = curve_fit(offset, T_data[fit_range_os_l[0]:fit_range_os_l[1]], R_data[fit_range_os_l[0]:fit_range_os_l[1]])
		popt_os_r, pcov_os_r = curve_fit(offset, T_data[fit_range_os_r[0]:fit_range_os_r[1]], R_data[fit_range_os_r[0]:fit_range_os_r[1]])

		R_p_10p = (popt_os_r[0] - popt_os_l[0]) * 0.1 + popt_os_l[0]
		search_points = R_data[fit_range_os_l[1]:fit_range_os_l[0]]
		R_10p = search_points[np.abs(search_points-R_p_10p).argmin()]
		Tc_10p = min(T_data[np.where(R_data==R_10p)])
		print("Tc @ 10% = ",Tc_10p)
			
		R_p_90p = (popt_os_r[0] - popt_os_l[0]) * 0.9 + popt_os_l[0]
		search_points = R_data[fit_range_os_r[1]:fit_range_os_r[0]]
		R_90p = search_points[np.abs(search_points-R_p_90p).argmin()]
		Tc_90p = min(T_data[np.where(R_data==R_90p)])
		print("Tc @ 90% = ",Tc_90p)

		T_data_B_0 = T_data
		R_data_B_0 = R_data
		Tc_10p_B_0 = Tc_10p
		Tc_90p_B_0 = Tc_90p
		R_10p_B_0 = R_10p
		R_90p_B_0 = R_90p
			
		fig = plt.figure(figsize=self.figsize, dpi=80).add_subplot(1, 1, 1)
		plt.plot(T_data, R_data, '.', color = "deepskyblue", markersize=self.markersize)
		T_data = np.linspace(T_data[0],T_data[-1],1000)
		plt.plot(T_data[fit_plot_range_os_l[0]:fit_plot_range_os_l[1]], offset(T_data[fit_plot_range_os_l[0]:fit_plot_range_os_l[1]], *popt_os_l), 'w--', markersize=self.markersize)
		plt.plot(T_data[fit_plot_range_os_r[0]:fit_plot_range_os_r[1]], offset(T_data[fit_plot_range_os_r[0]:fit_plot_range_os_r[1]], *popt_os_r), 'w--', markersize=self.markersize)
		plt.plot(Tc_10p, R_10p, 'w+', markersize=20)
		plt.plot(Tc_90p, R_90p, 'w+', markersize=20)
		plt.xlabel("Temperatur T / K")
		plt.ylabel(r"Nb Widerstand R / $\Omega$")
		#plt.title("Tc Plot für I = 0A")
		maximize()
		plt.savefig(self.export_folder+"Tc_plot_B0"+self.export_extension, bbox_inches='tight', dpi=self.dpi)
		#plt.show()


			
		print(80*"_"+"\n\nPlotting: Tc Plot für B = 259mT @ I = 8A")

		# fitting the function
		plot_range = [970, 1100]
		fit_range = [None, None]
		
		fit_range_os_l = [None, 75]
		fit_plot_range_os_l = [None, None]
		#fit_plot_range_os_l = fit_range_os_l
		
		fit_range_os_r = [95, None]
		fit_plot_range_os_r = [None, None]
		#fit_plot_range_os_r = fit_range_os_r
			
		data = dataSet[0]
		t_data = data['t']
		T_data = data['T']
		R_data = data['R_P_1']

		t_data = t_data[plot_range[0]:plot_range[1]]
		T_data = T_data[plot_range[0]:plot_range[1]]
		R_data = R_data[plot_range[0]:plot_range[1]]

		popt_os_l, pcov_os_l = curve_fit(offset, T_data[fit_range_os_l[0]:fit_range_os_l[1]], R_data[fit_range_os_l[0]:fit_range_os_l[1]])
		popt_os_r, pcov_os_r = curve_fit(offset, T_data[fit_range_os_r[0]:fit_range_os_r[1]], R_data[fit_range_os_r[0]:fit_range_os_r[1]])

		R_p_10p = (popt_os_r[0] - popt_os_l[0]) * 0.1 + popt_os_l[0]
		search_points = R_data[fit_range_os_l[1]:fit_range_os_l[0]]
		R_10p = search_points[np.abs(search_points-R_p_10p).argmin()]
		Tc_10p = min(T_data[np.where(R_data==R_10p)])
		print("Tc @ 10% = ",Tc_10p)
			
		R_p_90p = (popt_os_r[0] - popt_os_l[0]) * 0.9 + popt_os_l[0]
		search_points = R_data[fit_range_os_r[1]:fit_range_os_r[0]]
		R_90p = search_points[np.abs(search_points-R_p_90p).argmin()]
		Tc_90p = min(T_data[np.where(R_data==R_90p)])
		print("Tc @ 90% = ",Tc_90p)
		
		T_data_B_1 = T_data
		R_data_B_1 = R_data
		Tc_10p_B_1 = Tc_10p
		Tc_90p_B_1 = Tc_90p
		R_10p_B_1 = R_10p
		R_90p_B_1 = R_90p
		
		fig = plt.figure(figsize=self.figsize, dpi=80).add_subplot(1, 1, 1)
		plt.plot(T_data, R_data, '.', color="tab:green", markersize=self.markersize)
		T_data = np.linspace(T_data[0],T_data[-1],1000)
		plt.plot(T_data[fit_plot_range_os_l[0]:fit_plot_range_os_l[1]], offset(T_data[fit_plot_range_os_l[0]:fit_plot_range_os_l[1]], *popt_os_l), 'w--', markersize=self.markersize)
		plt.plot(T_data[fit_plot_range_os_r[0]:fit_plot_range_os_r[1]], offset(T_data[fit_plot_range_os_r[0]:fit_plot_range_os_r[1]], *popt_os_r), 'w--', markersize=self.markersize)
		plt.plot(Tc_10p, R_10p, 'w+', markersize=20)
		plt.plot(Tc_90p, R_90p, 'w+', markersize=20)
		plt.xlabel("Temperatur T / K")
		plt.ylabel(r"Nb Widerstand R / $\Omega$")
		#plt.title("Tc Plot für I = 0A")
		maximize()
		plt.savefig(self.export_folder+"Tc_plot_B1"+self.export_extension, bbox_inches='tight', dpi=self.dpi)
		#plt.show()



		print(80*"_"+"\n\nPlotting: Tc Plot für B = 129mT @ I = 4A")

		# fitting the function
		#plot_range = [1200, None]
		plot_range = [1240, -30]
		fit_range = [None, None]
		
		#fit_range_os_l = [-65, None]
		fit_range_os_l = [-38, None]
		fit_plot_range_os_l = [None, None]
		#fit_plot_range_os_l = fit_range_os_l
		
		#fit_range_os_r = [None, -80]
		fit_range_os_r = [None, -47]
		fit_plot_range_os_r = [None, None]
		#fit_plot_range_os_r = fit_range_os_r
			
		data = dataSet[0]
		t_data = data['t']
		T_data = data['T']
		R_data = data['R_P_1']

		t_data = t_data[plot_range[0]:plot_range[1]]
		T_data = T_data[plot_range[0]:plot_range[1]]
		R_data = R_data[plot_range[0]:plot_range[1]]

		popt_os_l, pcov_os_l = curve_fit(offset, T_data[fit_range_os_l[0]:fit_range_os_l[1]], R_data[fit_range_os_l[0]:fit_range_os_l[1]])
		popt_os_r, pcov_os_r = curve_fit(offset, T_data[fit_range_os_r[0]:fit_range_os_r[1]], R_data[fit_range_os_r[0]:fit_range_os_r[1]])

		R_p_10p = (popt_os_r[0] - popt_os_l[0]) * 0.1 + popt_os_l[0]
		search_points = R_data[fit_range_os_l[1]:fit_range_os_l[0]]
		R_10p = search_points[np.abs(search_points-R_p_10p).argmin()]
		Tc_10p = min(T_data[np.where(R_data==R_10p)])
		print("Tc @ 10% = ",Tc_10p)
			
		R_p_90p = (popt_os_r[0] - popt_os_l[0]) * 0.9 + popt_os_l[0]
		search_points = R_data[fit_range_os_r[1]:fit_range_os_r[0]]
		R_90p = search_points[np.abs(search_points-R_p_90p).argmin()]
		Tc_90p = min(T_data[np.where(R_data==R_90p)])
		print("Tc @ 90% = ",Tc_90p)
		
		T_data_B_2 = T_data
		R_data_B_2 = R_data
		Tc_10p_B_2 = Tc_10p
		Tc_90p_B_2 = Tc_90p
		R_10p_B_2 = R_10p
		R_90p_B_2 = R_90p
		
		fig = plt.figure(figsize=self.figsize, dpi=80).add_subplot(1, 1, 1)
		plt.plot(T_data, R_data, '.', color="tab:orange", markersize=self.markersize)
		T_data = np.linspace(T_data[0],T_data[-1],1000)
		plt.plot(T_data[fit_plot_range_os_l[0]:fit_plot_range_os_l[1]], offset(T_data[fit_plot_range_os_l[0]:fit_plot_range_os_l[1]], *popt_os_l), 'w--', markersize=self.markersize)
		plt.plot(T_data[fit_plot_range_os_r[0]:fit_plot_range_os_r[1]], offset(T_data[fit_plot_range_os_r[0]:fit_plot_range_os_r[1]], *popt_os_r), 'w--', markersize=self.markersize)
		plt.plot(Tc_10p, R_10p, 'w+', markersize=20)
		plt.plot(Tc_90p, R_90p, 'w+', markersize=20)
		plt.xlabel("Temperatur T / K")
		plt.ylabel(r"Nb Widerstand R / $\Omega$")
		#plt.title("Tc Plot für I = 0A")
		maximize()
		plt.savefig(self.export_folder+"Tc_plot_B2"+self.export_extension, bbox_inches='tight', dpi=self.dpi)
		plt.show()



		print(80*"_"+"\n\nPlotting: Tc Plot (alle)")

		
		fig = plt.figure(figsize=self.figsize, dpi=80).add_subplot(1, 1, 1)
		plt.plot(T_data_B_0, R_data_B_0, '.', color = "deepskyblue", markersize=self.markersize, label = r"$\mu_0$H=0")
		plt.plot(T_data_B_2, R_data_B_2, '.', color = "tab:orange", markersize=self.markersize, label = r"$\mu_0$H=129mT")
		plt.plot(T_data_B_1, R_data_B_1, '.', color = "tab:green", markersize=self.markersize, label = r"$\mu_0$H=259mT")
		plt.xlabel("Temperatur T / K")
		plt.ylabel(r"Nb Widerstand R / $\Omega$")
		plt.xlim(5, None)
		#plt.title("Tc Plot für I = 0A")
		#plt.legend(loc='lower right')
		plt.legend(loc='upper left', markerscale=2)
		maximize()
		plt.savefig(self.export_folder+"Tc_plot_all"+self.export_extension, bbox_inches='tight', dpi=self.dpi)
		plt.show()


			
		print(80*"_"+"\n\nPlotting: Phasendiagramm Niob")
		
		# fitting the function
		fit_range = [None, None]
		fit_plot_range = [None, None]
		
		I_data = [0, 4, 8]
		B_data = [i * 194/6 for i in I_data]	# 6A entspricht 194mT
		
		phi0 = 2.067833848e-15
		
		# for 10percent curve
		T_data_10p = [Tc_10p_B_0, Tc_10p_B_2, Tc_10p_B_1]
		
		popt_10p, pcov_10p = curve_fit(lin, T_data_10p[fit_range[0]:fit_range[1]], B_data[fit_range[0]:fit_range[1]]) 
			
		Tc = Tc_10p_B_0
		S = 1e-3*popt_10p[1]
		xi0 = np.sqrt(-phi0/(2 * np.pi * S * Tc))
		l = xi0**2 / 39e-9
		B_c2 = phi0 / (2 * np.pi * xi0**2)
		print("\nFür 10%:\n")
		print("Kohärenzlänge xi0={:.4}nm".format(1e9*xi0))
		print("mittlere freie Weglänge l={:.4}nm".format(1e9*l))
		print("oberes krit. Magnetfeld Bc2={:.4}T".format(B_c2))
		
		# for 90percent curve
		T_data_90p = [Tc_90p_B_0, Tc_90p_B_2, Tc_90p_B_1]
		
		popt_90p, pcov_90p = curve_fit(lin, T_data_90p[fit_range[0]:fit_range[1]], B_data[fit_range[0]:fit_range[1]])
			
		Tc = Tc_90p_B_0
		S = 1e-3*popt_90p[1]
		xi0 = np.sqrt(-phi0/(2 * np.pi * S * Tc))
		l = xi0**2 / 39e-9
		B_c2 = phi0 / (2 * np.pi * xi0**2)
		print("\nFür 90%:\n")
		print("Kohärenzlänge xi0={:.4}nm".format(1e9*xi0))
		print("mittlere freie Weglänge l={:.4}nm".format(1e9*l))
		print("oberes krit. Magnetfeld Bc2={:.4}T".format(B_c2))
		
		fig = plt.figure(figsize=self.figsize, dpi=80).add_subplot(1, 1, 1)
		plt.plot(T_data_10p, B_data, 'o', color = "deepskyblue")
		plt.plot(T_data_90p, B_data, 'o', color = "tab:orange")
		T_data_10p = np.linspace(T_data_10p[fit_plot_range[0]:fit_plot_range[1]][0],T_data_10p[fit_plot_range[0]:fit_plot_range[1]][-1],1000)
		T_data_90p = np.linspace(T_data_90p[fit_plot_range[0]:fit_plot_range[1]][0],T_data_90p[fit_plot_range[0]:fit_plot_range[1]][-1],1000)
		plt.plot(T_data_10p, lin(T_data_10p, *popt_10p), '--', markersize=self.markersize, color= "deepskyblue", label = r"Lin. Fit: $B_C=S\cdot T_C+a$"+"\n"+r"S=({:.4}$\pm${:.3})mT/K".format(popt_10p[1],np.sqrt(np.diag(pcov_10p))[1]))
		plt.plot(T_data_90p, lin(T_data_90p, *popt_90p), '--', markersize=self.markersize, color= "tab:orange", label = r"Lin. Fit: $B_C=S\cdot T_C+a$"+"\n"+r"S=({:.4}$\pm${:.3})mT/K".format(popt_90p[1],np.sqrt(np.diag(pcov_90p))[1]))
		plt.xlabel(r"krit. Temperatur $T_C$ / K")
		plt.ylabel(r"krit. Magnetfeld $B_C$ / mT")
		plt.xlim(None,10.5)
		plt.ylim(0,None)
		#plt.title("Phasendiagramm Niob")
		leg = plt.legend(loc='upper right')
		#leg.get_frame().set_edgecolor('w')
		#leg.get_frame().set_facecolor('none')
		maximize()
		plt.savefig(self.export_folder+"phasen_diagramm"+self.export_extension, bbox_inches='tight', dpi=self.dpi)
		plt.show()