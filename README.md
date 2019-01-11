# Histcon README
Thyroid Project Guide

A. Files
 -------------------------------------------------------------------------------
| Large Storage Volume, 4TB (Other files/Thyroid)				|
|										|
| Folders									|
| /512			# 512 x 512px, height/width of 140um			|
| /512_from_1024	# 512 x 512px, height/width of 280um			|
| /1024			# 1024 x 1024px, height/width of 280um			|
| /Finished_SVS		# Raw *.svs files, already annotated			|
| /Annotations		# Annotated JPG chunks, in *.jpg and *.json format	|
|										|
 -------------------------------------------------------------------------------
| Small storage volume, 512GB (~/histcon)					|
| 										|
| Folders									|
| /full_images		# JPG whole-slide images				|
| /half_images		# JPG whole-slide images down-scaled 50%		|
| /model		# Tensorflow & Tensorboard model & training output	|
| /train_data		# Neural net training data, split into sub-folders	|
| /convoluter.py	# Applies a finished model to a whole-slide image	|
| /histcon.py		# Train and save a new model using train_data images	|
 -------------------------------------------------------------------------------


B. Useful commands

# Using nConvert to create thumbnails @ 10% (for use with LabelMe):
nconvert -out jpeg -o %_T.jpg -resize 10% 10% *.jpg

# Resize 1024 -> 512
nconvert -out jpeg -o resized/%.jpg -resize 50% 50% *.jpg

# List all files of size 512
find -iname "*.jpg" -exec identify {} \; | grep 512

#Run tensorboard
python3 ~/.local/lib/python3.6/site-packages/tensorboard/main.py --logdir=~/histcon/model



