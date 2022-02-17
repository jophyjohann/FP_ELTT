from dataset_operations2 import DataSet_Operations2
import matplotlib.pyplot as plt
import numpy as np
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

		### Plot all measured datasets (14) and export them for overwiev ###

		for data in dataSet:
			export_name = data['name'][:5] + data['name'][-7:-4]
			name = ("dataSet[" + str(dataSet.index(data)) + "]\n" + export_name).replace("_"," ")
			title_name=export_name.replace("_"," ")

			print(80*"_"+"\n\nPlotting: ", name)
			measured_datas = [["time t / s", "temperature T / K", r"sample 1 resistance $R_1$ / $\Omega$", r"Therm. resistance $R_T$ / $\Omega$", r"sample 2 resistance $R_2$ / $\Omega$"],
												[data['t'], data['T'], data['R_P_1'], data['R_T'], data['R_P_2']]]
			
			for i in range(1,len(measured_datas[1])):
				
				measurement = measured_datas[1][i]
				label = measured_datas[0][i]
				title = title_name + " (" + label + ")"
				fig = plt.figure(figsize=(8, 4), dpi=120).add_subplot(1, 1, 1)
				plt.plot(measured_datas[1][0], measurement, '-', label = label)
				plt.title(label = title)
				plt.xlabel(measured_datas[0][0])
				plt.ylabel(title[10:-1])
				plt.xlim(0,None)
				plt.ylim(0,None)
				plt.legend()
				maximize()
			plt.show()