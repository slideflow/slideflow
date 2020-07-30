import os
import argparse

import slideflow.util as sfutil

from os.path import join, isdir, exists

def update(root):
	# Updates slideflow projects from version 1.6-1.8 to 1.9
	project_folders = [f for f in os.listdir(root) if isdir(join(root, f)) and exists(join(root, f, 'settings.json'))]
	for folder in project_folders:
		project_folder = join(root, folder)
		project_settings = sfutil.load_json(join(project_folder, 'settings.json'))
		if 'tile_px' in project_settings:
			tile_px = project_settings['tile_px']
			tile_um = project_settings['tile_um']
			if exists(join(project_folder, 'models')):
				model_folders = [m for m in os.listdir(join(project_folder, 'models')) if isdir(join(project_folder, 'models', m))]
				for model in model_folders:
					model_folder = join(project_folder, 'models', model)
					hp_file = join(model_folder, 'hyperparameters.json')
					if exists(hp_file):
						hp = sfutil.load_json(hp_file)
						if 'tile_px' not in hp['hp']:
							hp['hp']['tile_px'] = tile_px
							hp['hp']['tile_um'] = tile_um
						sfutil.write_json(hp, hp_file)
						print(f"Updated model {model} in project {folder}")

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = "Update utility (1.8 -> 1.9)")
	parser.add_argument('-pr', required=True, help='Path to root directory containing projects.')
	args = parser.parse_args()
	update(args.pr)