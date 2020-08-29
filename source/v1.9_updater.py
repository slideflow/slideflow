import os
import argparse
import shutil
import slideflow.util as sfutil

from os.path import join, isdir, exists

def update(root):
	'''Updates slideflow projects from version 1.6-1.8 to 1.9'''
	project_folders = [f for f in os.listdir(root) if isdir(join(root, f)) and exists(join(root, f, 'settings.json'))]
	for folder in project_folders:
		project_folder = join(root, folder)
		project_settings = sfutil.load_json(join(project_folder, 'settings.json'))
		# Look for projects built with 1.8 or earlier
		if 'tile_px' in project_settings:
			# Move tile_px and tile_um from project settings.json into model hyperparameters files
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
			# Scan datasets to ensure dataset organization follows 1.9 labeling conventions, renaming accordingly
			if 'dataset_config' not in project_settings:
				print(f"Warning: unable to update old (v1.3 or earlier) project at {project_folder}")
				continue
			dataset_config_file = project_settings['dataset_config']
			if not exists(dataset_config_file): continue
			shutil.copy(dataset_config_file, dataset_config_file+'.backup')
			dataset_config = sfutil.load_json(dataset_config_file)
			project_datasets = project_settings['datasets']
			for dataset in project_datasets:
				if dataset in dataset_config:
					label = dataset_config[dataset]['label']
					new_label = f'{tile_px}px_{tile_um}um'
					if label != new_label:
						# Update TFRecord label
						if exists(join(dataset_config[dataset]['tfrecords'], new_label)):
							print(f"Unable to auto-update dataset {dataset} tfrecords labeling automatically at {dataset_config[dataset]['tfrecords']}, please ensure dataset label matches {new_label}")
						elif exists(join(dataset_config[dataset]['tfrecords'], label)):
							shutil.move(join(dataset_config[dataset]['tfrecords'], label), join(dataset_config[dataset]['tfrecords'], new_label))
							print(f"Moved tfrecords in dataset {dataset} into new label directory, {new_label}")
						# Update Tiles label
						if exists(join(dataset_config[dataset]['tiles'], new_label)):
							print(f"Unable to auto-update dataset {dataset} tiles labeling automatically at {dataset_config[dataset]['tfrecords']}, please ensure dataset label matches {new_label}")
						elif exists(join(dataset_config[dataset]['tiles'], label)):
							shutil.move(join(dataset_config[dataset]['tiles'], label), join(dataset_config[dataset]['tiles'], new_label))
							print(f"Moved tiles in dataset {dataset} into new label directory, {new_label}")
					dataset_config[dataset]['label'] = new_label
			sfutil.write_json(dataset_config, dataset_config_file)
			print(f"Completed update of project {sfutil.bold(project_settings['name'])} using dataset configuration JSON at {sfutil.green(dataset_config_file)}")

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = "Update utility (1.8 -> 1.9)")
	parser.add_argument('-pr', required=True, help='Path to root directory containing projects.')
	parser.add_argument('--nested', action="store_true", help='Whether to search recursively through a parent directory into nested sub-directories.')
	args = parser.parse_args()
	if not args.nested:
		update(args.pr)
	else:
		nested_folders = [f for f in os.listdir(args.pr) if isdir(join(args.pr, f))]
		for nested_folder in nested_folders:
			update(join(args.pr, nested_folder))
