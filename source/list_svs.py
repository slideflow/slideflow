import os
import argparse

from shutil import copy
from os.path import exists, isfile, join

def copy_files(file_list, destination):
	for _file in file_list:
		copy(_file, destination)

def delete_files(file_list):
	for _file in file_list:
		os.remove(_file)

def main(_dir):
	compare_dir = '/home/shawarma/data/slideflow_projects/tcga_thca_follicular/tiles/train_data2'
	svs = []
	for subdir in os.listdir(_dir):
		sub_svs = [join(_dir, subdir, i) for i in os.listdir(join(_dir, subdir)) if (isfile(join(_dir, subdir, i)) and (i[-3:].lower() == "svs"))]	
		svs += sub_svs
	svs_names = [i.split('/')[-1][:-4] for i in svs]
	for i in svs_names: 
		tag = " found" if i in os.listdir(compare_dir) else " not found"
		print(i + tag)	

if __name__=="__main__":
	parser = argparse.ArgumentParser(description = "Lists SVS files in a directory.")
	parser.add_argument('-d', '--dir', default="/media/shawarma/Backup/Other_files/TCGA/THCA", help='Path to SVS directory.')
	args = parser.parse_args()
	main(args.dir)