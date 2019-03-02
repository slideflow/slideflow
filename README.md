# Histcon README
Thyroid Project Guide

A. Files

... automated docs to be added.

B. Useful commands

# Using nConvert to create thumbnails @ 10% (for use with LabelMe):
nconvert -out jpeg -o %_T.jpg -resize 10% 10% *.jpg

# Resize 1024 -> 512
nconvert -out jpeg -o resized/%.jpg -resize 50% 50% *.jpg

# List all files of size 512
find -iname "*.jpg" -exec identify {} \; | grep 512

# Run tensorboard
python3 ~/.local/lib/python3.6/site-packages/tensorboard/main.py --logdir=~/histcon/model

# Activate tensorflow environment
source ./venv/bin/activate

# Copy large number of images
find ~/histcon/packing_results/normal/ -name '*.jpg' -exec cp -t ~/histcon/train_data/0 {} +


