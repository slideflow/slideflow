import os
from os.path import isfile, join
import subprocess

root_folder = "/home/shawarma/thyroid/images/Five_Category_512_from_2048/to_resize"
resize_name = "512_from_2048"
resize_folder = join(root_folder, resize_name)
if not os.path.isdir(resize_folder): 
	os.mkdir(resize_folder)

cases = [f for f in os.listdir(root_folder) if (not isfile(join(root_folder, f)) and f != resize_name)]
for case in cases:
	print("Converting images from case {}".format(case))
	os.mkdir(join(resize_folder, case))
	command = 'nconvert -out jpeg -o {}/%.jpg -resize 25% 25% {}/*.jpg'.format(join(resize_folder, case), join(root_folder, case))
	process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
	