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

		data = dataSet[1]

		plot_lims = [[700,1000],
								[1150,1450],
								[1500,1750],
								[2200,2450],
								[2800,3050],
								[3200,3500],
								[3700,4050]]
			
		
		# set plot characteristics of x and y ticks and grid
		major_x_ticks = 500
		minor_x_ticks = 100
		major_y_ticks = 0.005
		minor_y_ticks = 0.001
		major_alpha = 0.7
		minor_alpha = 0.2

		fig = plt.figure(figsize=(8, 4), dpi=120).add_subplot(1, 1, 1)
		plt.plot(data['t'], data['R_P_1'], '-')
		plt.xlabel("T / K")
		plt.ylabel(r"R / $\Omega$")
		plt.xlim(0,None)
		plt.ylim(0,None)
		fig.set_xticks(np.arange(plt.xlim()[0],plt.xlim()[1],major_x_ticks))
		fig.set_xticks(np.arange(plt.xlim()[0],plt.xlim()[1],minor_x_ticks),minor=True)
		fig.set_yticks(np.arange(plt.ylim()[0],plt.ylim()[1],major_y_ticks))
		fig.set_yticks(np.arange(plt.ylim()[0],plt.ylim()[1],minor_y_ticks),minor=True)
		fig.grid(which='minor', alpha=minor_alpha)
		fig.grid(which='major', alpha=major_alpha)
		maximize()
		plt.show()

			
		print(80*"_"+"\n\nPlotting: Tc Plot für I = 0A")
			
		fig = plt.figure(figsize=(8, 4), dpi=120).add_subplot(1, 1, 1)
		plt.plot(data['t'], data['R_P_1'], '-')
		plt.xlabel("T / K")
		plt.ylabel(r"R / $\Omega$")
		plt.xlim(plot_lims[0][0],plot_lims[0][1])
		#plt.ylim(0,None)
		fig.set_xticks(np.arange(plt.xlim()[0],plt.xlim()[1],major_x_ticks))
		fig.set_xticks(np.arange(plt.xlim()[0],plt.xlim()[1],minor_x_ticks),minor=True)
		fig.set_yticks(np.arange(plt.ylim()[0],plt.ylim()[1],major_y_ticks))
		fig.set_yticks(np.arange(plt.ylim()[0],plt.ylim()[1],minor_y_ticks),minor=True)
		fig.grid(which='minor', alpha=minor_alpha)
		fig.grid(which='major', alpha=major_alpha)
		maximize()
		plt.show()

		