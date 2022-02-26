import glob, os
import numpy as np

class DataSet_Operations:
	def __init__(self):
		self.folder_name="measurements/"
		self.file_extension=".dat"

		self.this_file = __name__ + ".py"
		self.working_path = os.getcwd()

		#automatically inserted dataset_files:
		self.dataset_files = ["Heinzelmann_Vincent_Nb-Si.dat",
										 "Heinzelmann_Vincent_Nb_H-Feld.dat",
										 "Heinzelmann_Vincent_Cu-Si.dat",
										 ]
		#end of automatically inserted dataset_files


	def datasets_change_comma_to_dot(self):
		for filename in self.dataset_files:
			#print(filename)
			file =  open(self.folder_name + filename,"r")
			data =file.read().replace(",", ".")
			file = open(self.folder_name + filename,"w")
			file.write(data)
			file.close()


	def insert_dataset_files(self):
		os.chdir(self.folder_name)
		new_filenames = glob.glob("*" + self.file_extension)
		os.chdir(self.working_path)

		filename_list = '['
		for file in new_filenames:
			filename_list += '"' + file + '",\n										 '
		filename_list += ']'
		filename_list.replace(",\n										 ]","]")
		filename_list += "\n"

		file = open(self.this_file,"r")
		data = file.read()
		file = open(self.this_file,"w")
		start_string = "		#automatically inserted dataset_files:"
		stop_string = "		#end of automatically inserted dataset_files"
		additional_string = "\n		self.dataset_files = "
		dataset_spot=data[data.find(start_string):data.find(stop_string)] + stop_string
		file.write(data.replace(dataset_spot, start_string + additional_string + filename_list + stop_string))
		file.close()


	def import_dataset_measurements(self):
		self.dataSet = [None] * len(self.dataset_files)
		for i in range(len(self.dataset_files)):
			self.dataSet[i] = {
        	'name': self.dataset_files[i],
					't': np.loadtxt(self.folder_name + self.dataset_files[i], unpack=True, comments="#", usecols=(0), delimiter="\t"),
        	'T': np.loadtxt(self.folder_name + self.dataset_files[i], unpack=True, comments="#", usecols=(1), delimiter="\t"),
        	'R_P_1': np.loadtxt(self.folder_name + self.dataset_files[i], unpack=True, comments="#", usecols=(2), delimiter="\t"),
    			'R_T': np.loadtxt(self.folder_name + self.dataset_files[i], unpack=True, comments="#", usecols=(3), delimiter="\t"),
    			'R_P_2': np.loadtxt(self.folder_name + self.dataset_files[i], unpack=True, comments="#", usecols=(4), delimiter="\t"),
    			}
