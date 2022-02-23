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

		def lin(x, a, S):
			#return (1 - (T / Tc)) * phi0 / (2 * np.pi * xi0**2)
			return S * x + a
		
		def offset(x, b):
			x = [b for i in x]
			return x
		
		print(80*"_"+"\n\nSupraleitung Auswertung2 eigene Messdaten")

		
		print(80*"_"+"\n\nPlotting: Tc Plot für I = 0A")

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


			
		print(80*"_"+"\n\nPlotting: Tc Plot für I = 1A")

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


			
		print(80*"_"+"\n\nPlotting: Tc Plot für I = 2A")

		# fitting the function
		plot_range = [670, 770]
		fit_range = [None, None]
		
		fit_range_os_l = [-50, None]
		fit_plot_range_os_l = [None, None]
		#fit_plot_range_os_l = fit_range_os_l
		
		fit_range_os_r = [None, -65]
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


			
		print(80*"_"+"\n\nPlotting: Tc Plot für I = 3A")

		# fitting the function
		plot_range = [970, 1100]
		fit_range = [None, None]
		
		fit_range_os_l = [None, 75]
		fit_plot_range_os_l = [None, None]
		#fit_plot_range_os_l = fit_range_os_l
		
		fit_range_os_r = [95, None]
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



		print(80*"_"+"\n\nPlotting: Tc Plot für I = 4A")

		# fitting the function
		plot_range = [1200, None]
		fit_range = [None, None]
		
		fit_range_os_l = [-65, None]
		fit_plot_range_os_l = [None, None]
		#fit_plot_range_os_l = fit_range_os_l
		
		fit_range_os_r = [None, -80]
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
