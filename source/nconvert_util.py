import os
from os.path import isfile, join
import subprocess
import argparse
from PIL import Image

def main(root_folder, size, augment):
	resize_name = "resized"
	resize_folder = join(root_folder, resize_name)
	if not os.path.isdir(resize_folder): 
		os.mkdir(resize_folder)

	cases = [f for f in os.listdir(root_folder) if (not isfile(join(root_folder, f)) and f != resize_name)]
	processes = []

	# Resize all images
	for case in cases:
		print("Converting images from case {}".format(case))
		os.mkdir(join(resize_folder, case))
		command = 'nconvert -out jpeg -o {}/%.jpg -resize {} {} {}/*.jpg'.format(join(resize_folder, case), size, size, join(root_folder, case))
		processes.append(subprocess.Popen(command, shell=True, stdout=subprocess.PIPE))

	# Wait for prior step to finish
	for process in processes:
		process.wait()

	# Generate flipped/rotated images (data augmentation)
	if augment:
		for case in cases:
			print("Augmenting data from case {}".format(case))
			export_dir = join(resize_folder, case)
			case_images = [f for f in os.listdir(export_dir) if (isfile(join(export_dir, f)) and (f[-3:] == 'jpg'))]
			for case_im_name in case_images:
				print(" > Augmenting image {}".format(case_im_name))
				case_im = Image.open(join(export_dir, case_im_name))
				case_im.transpose(Image.FLIP_LEFT_RIGHT).save(join(export_dir, "%s_0F.jpg" % case_im_name), "jpeg")
				case_im.transpose(Image.ROTATE_90).save(join(export_dir, "%s_90.jpg" % case_im_name), "jpeg")
				case_im.transpose(Image.ROTATE_90).transpose(Image.FLIP_LEFT_RIGHT).save(join(export_dir, "%s_90F.jpg" % case_im_name), "jpeg")
				case_im.transpose(Image.ROTATE_180).save(join(export_dir, "%s_180.jpg" % case_im_name), "jpeg")
				case_im.transpose(Image.ROTATE_180).transpose(Image.FLIP_LEFT_RIGHT).save(join(export_dir, "%s_180F.jpg" % case_im_name), "jpeg")
				case_im.transpose(Image.ROTATE_270).save(join(export_dir, "%s_270.jpg" % case_im_name), "jpeg")
				case_im.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT).save(join(export_dir, "%s_270F.jpg" % case_im_name), "jpeg")
				case_im.close()
			
if __name__==('__main__'):
	parser = argparse.ArgumentParser(description = "Tool to batch convert images from a directory of cases. ")
	parser.add_argument('-d', '--dir', help='Path to root directory containing case directories with images to resize.')
	parser.add_argument('-s', '--size', type=int, help='Pixel size for which to resize images.')
	parser.add_argument('--augment', action="store_true", help='Flag to indicate script should perform data augmentation (flipping/rotating)')
	args = parser.parse_args()

	main(args.dir, args.size, args.augment)