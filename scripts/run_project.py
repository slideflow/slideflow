'''Utility script for executing Slideflow project functions.

For easier experiment tracking, Slideflow projects include an `actions.py`
file in their root directory, with the following structure:

    def main(P):
        P.do_something()

This file can be modified to organize a complete experiment. For example, to
set up an experiment which extracts tiles from slides then trains a model
to the outcome variable 'is_tumor', modify the file as follows:

    def main(P):
        import slideflow as sf

        P.extract_tiles(tile_px=299, tile_um=302)
        hp = sf.ModelParams(
            tile_px=299,
            tile_um=302,
            model='xception',
            learning_rate=0.0001,
            ...
        )
        P.train(
            'is_tumor',
            params=hp
        )

Then execute the functions in this file using this `run_project.py` script:

    python3 run_project.py -p /path/to/project_folder

This functionality is entirely optional, as Slideflow can also be used as
a regular package in scripts, Jupyter notebooks, or an interactive shell.
'''

import os
import sys
import slideflow as sf
import argparse
import logging
import multiprocessing


if __name__=='__main__':
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-p', '--project', required=True, help='Path to project directory.'
    )
    parser.add_argument(
        '-n', '--neptune', action="store_true", help="Use Neptune logger."
    )
    parser.add_argument(
        '--nfs',
        action="store_true",
        help="Sets environmental variable HDF5_USE_FILE_LOCKING='FALSE' as a "
             "fix to problems with NFS file systems."
    )
    parser.add_argument(
        '--debug',
        action="store_true",
        help="Increase verbosity of logging output to include debug messages."
    )
    args = parser.parse_args()
    if args.nfs:
        os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
        print("Set environmental variable 'HDF5_USE_FILE_LOCKING'='FALSE'")
    if args.debug:
        sf.setLoggingLevel(logging.DEBUG)

    sf.about()
    P = sf.Project.from_prompt(args.project, use_neptune=args.neptune)
    P.associate_slide_names()

    sys.path.insert(0, args.project)
    try:
        import actions
    except Exception as e:
        print(f"Error loading actions.py in {args.project}; either does not"
              " exist or contains an error")
        raise e

    actions.main(P)
