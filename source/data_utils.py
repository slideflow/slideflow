import shutil

from os import listdir, makedirs
from os.path import isfile, isdir, join, exists
from random import shuffle
import argparse

def make_dir(_dir):
	''' Makes a directory if one does not already exist, in a manner compatible with multithreading. '''
	if not exists(_dir):
		try:
			makedirs(_dir, exist_ok=True)
		except FileExistsError:
			pass

def build_validation(train_dir, eval_dir, fraction = 0.1):
	make_dir(eval_dir)
	cat_dirs = [_dir for _dir in listdir(train_dir) if isdir(join(train_dir, _dir))]
	for cat_dir in cat_dirs:
		print("Category {}:".format(cat_dir))
		make_dir(join(eval_dir, cat_dir))
		case_dirs = [_dir for _dir in listdir(join(train_dir, cat_dir)) if isdir(join(train_dir, cat_dir, _dir))]
		for case_dir in case_dirs:
			make_dir(join(eval_dir, cat_dir, case_dir))
			files = [file for file in listdir(join(train_dir, cat_dir, case_dir)) 
						if (isfile(join(train_dir, cat_dir, case_dir, file))) and
						   (file[-3:] == "jpg")]
			shuffle(files)
			num_to_move = int(len(files)*fraction)
			for file in files[0:num_to_move]:
				shutil.move(join(train_dir, cat_dir, case_dir, file), join(eval_dir, cat_dir, case_dir, file))
			print("  Moved {} files for case {}".format(num_to_move, case_dir))

def merge_validation(train_dir, eval_dir):
	cat_dirs = [_dir for _dir in listdir(eval_dir) if isdir(join(eval_dir, _dir))]
	for cat_dir in cat_dirs:
		print("Category {}:".format(cat_dir))
		case_dirs = [_dir for _dir in listdir(join(eval_dir, cat_dir)) if isdir(join(eval_dir, cat_dir, _dir))]
		for case_dir in case_dirs:
			files = [file for file in listdir(join(eval_dir, cat_dir, case_dir)) 
						if (isfile(join(eval_dir, cat_dir, case_dir, file))) and
						   (file[-3:] == "jpg")]
			for file in files:
				shutil.move(join(eval_dir, cat_dir, case_dir, file), join(train_dir, cat_dir, case_dir, file))
			print("  Merged {} files for case {}".format(len(files), case_dir))

if __name__==('__main__'):
	parser = argparse.ArgumentParser(description = 'Tool to build and re-merge a validation dataset from an existing training dataset. Training dataset must be in a folder called "train_data".')
	parser.add_argument('-d', '--dir', help='Path to root directory containing "train_data".')
	parser.add_argument('-f', '--fraction', type=float, default = 0.1, help='Fraction of training dataset to use for validation (default = 0.1).')
	parser.add_argument('--build', action="store_true", help='Build a new validation dataset by extraction a certain percentage of images from the training dataset.')
	parser.add_argument('--merge', action="store_true", help='Merge an existing validation dataset ("eval_data" directory) into an existing "train_data" directory.')
	args = parser.parse_args()

	train_dir = join(args.dir, "train_data")
	eval_dir = join(args.dir, "eval_data")

	if args.build:
		build_validation(train_dir, eval_dir, fraction = args.fraction)
	elif args.merge:
		merge_validation(train_dir, eval_dir)
	else:
		print("Error: you must specify either a '--build' or '--merge' flag.")