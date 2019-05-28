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
	svs = []
	for subdir in os.listdir(_dir):
		sub_svs = [join(_dir, subdir, i) for i in os.listdir(join(_dir, subdir)) if (isfile(join(_dir, subdir, i)) and (i[-6:].lower() == "qpdata"))]	
		svs += sub_svs
	#delete_files(svs)

if __name__=="__main__":
	parser = argparse.ArgumentParser(description = "Lists SVS files in a directory.")
	parser.add_argument('-d', '--dir', default="D:\Other_files\TCGA\THCA", help='Path to SVS directory.')
	args = parser.parse_args()
	main(args.dir)