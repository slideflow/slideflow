import os
import argparse
import shutil
import slideflow as sf

from os.path import join, isdir, exists

#TODO: update validation logs ("outcome" -> "outcome_label")

def update_project_models(project_folder):
    import tensorflow as tf
    from slideflow.model.utils import negative_log_likelihood, concordance_index
    folder = project_folder

    if exists(join(project_folder, 'models')):
        model_folders = [mf for mf in os.listdir(join(project_folder, 'models')) if isdir(join(project_folder, 'models', mf))]
        for model_folder in model_folders:
            full_model_folder = join(project_folder, 'models', model_folder)
            hyperparameters = sf.util.get_model_params(full_model_folder)
            if hyperparameters is None:
                print(f"Unable to find hyperparameters file for model {folder} > {model_folder}, skipping")
                continue

            models = [m for m in os.listdir(full_model_folder) if sf.util.path_to_ext(m) == 'h5']
            for model in models:
                model_path = join(full_model_folder, model)
                new_model_path = join(full_model_folder, sf.util.path_to_name(model))

                print(f"Upgrading {sf.util.blue(folder)} > {sf.util.yellow(model_folder)} > {sf.util.green(model)} ... ", end="")
                try:
                    if hyperparameters['model_type'] == 'cph':
                        loaded_model = tf.keras.models.load_model(model_path,custom_objects = {
                                                                                'negative_log_likelihood':negative_log_likelihood,
                                                                                'concordance_index':concordance_index
                                                                            })
                    else:
                        loaded_model = tf.keras.models.load_model(model_path)
                    loaded_model.save(new_model_path)
                    os.remove(model_path)
                    print(sf.util.green('DONE'))
                except ValueError:

                    print(sf.util.red('FAIL'))
                    print(" - Unable to load model, incorrect python version")


def update_models(root):
    '''Updates models from Keras H5 to Tensorflow SavedModel format'''
    print(f"{sf.util.yellow('WARNING!!! ')} Although tested, this conversion function does not guarantee model integrity post-conversion.")
    print("Please backup your models before continuing!")
    input("Acknowledge (press enter) > ")
    print("Updating legacy models...")
    print(f"{sf.util.blue('PROJECT')} > {sf.util.yellow('MODEL_FOLDER')} > {sf.util.green('MODEL')}")

    project_folders = [f for f in os.listdir(root) if isdir(join(root, f)) and exists(join(root, f, 'settings.json'))]
    for folder in project_folders:
        project_folder = join(root, folder)
        update_project_models(project_folder)

def update_version(root):
    '''Updates slideflow projects from version 1.6-1.8 to 1.9+'''
    print("Updating project versions.")
    project_folders = [f for f in os.listdir(root) if isdir(join(root, f)) and exists(join(root, f, 'settings.json'))]
    for folder in project_folders:
        project_folder = join(root, folder)
        project_settings = sf.util.load_json(join(project_folder, 'settings.json'))
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
                        hp = sf.util.load_json(hp_file)
                        if 'tile_px' not in hp['hp']:
                            hp['hp']['tile_px'] = tile_px
                            hp['hp']['tile_um'] = tile_um
                            sf.util.write_json(hp, hp_file)
                            print(f"Updated model {model} in project {folder}")
            # Scan datasets to ensure dataset organization follows 1.9 labeling conventions, renaming accordingly
            if 'dataset_config' not in project_settings:
                print(f"Warning: unable to update old (v1.3 or earlier) project at {project_folder}")
                continue
            dataset_config_file = project_settings['dataset_config']
            if not exists(dataset_config_file): continue
            shutil.copy(dataset_config_file, dataset_config_file+'.backup')
            dataset_config = sf.util.load_json(dataset_config_file)
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
            sf.util.write_json(dataset_config, dataset_config_file)
            print(f"Completed update of project {sf.util.bold(project_settings['name'])} using dataset configuration JSON at {sf.util.green(dataset_config_file)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Slideflow update utility")
    parser.add_argument('-pr', required=True, help='Path to root directory containing projects.')
    parser.add_argument('--nested', action="store_true", help='Whether to search recursively through a parent directory into nested sub-directories.')
    parser.add_argument('--version', action="store_true", help='Upgrades projects from 1.6-1.8 to 1.9+')
    parser.add_argument('--models', action="store_true", help='Upgrades models from Keras H5 to Tensorflow SavedModel format')
    args = parser.parse_args()
    if not args.nested:
        if args.version:
            update_version(args.pr)
        if args.models:
            update_models(args.pr)
    else:
        nested_folders = [f for f in os.listdir(args.pr) if isdir(join(args.pr, f))]
        for nested_folder in nested_folders:
            if args.version:
                update_version(join(args.pr, nested_folder))
            if args.models:
                update_models(join(args.pr, nested_folder))
