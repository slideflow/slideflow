import os
from os.path import isfile, join
import subprocess
import argparse

def main(root_folder, size):
	resize_name = "resized"
	resize_folder = join(root_folder, resize_name)
	if not os.path.isdir(resize_folder): 
		os.mkdir(resize_folder)

	cases = [f for f in os.listdir(root_folder) if (not isfile(join(root_folder, f)) and f != resize_name)]
	for case in cases:
		print("Converting images from case {}".format(case))
		os.mkdir(join(resize_folder, case))
		command = 'nconvert -out jpeg -o {}/%.jpg -resize {} {} {}/*.jpg'.format(join(resize_folder, case), size, size, join(root_folder, case))
		process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
		
if __name__==('__main__'):
	parser = argparse.ArgumentParser(description = "Tool to batch convert images from a directory of cases. ")
	parser.add_argument('-d', '--dir', help='Path to root directory containing case directories with images to resize.')
	parser.add_argument('-s', '--size', type=int, help='Pixel size for which to resize images.')
	args = parser.parse_args()

	main(args.dir, args.size)