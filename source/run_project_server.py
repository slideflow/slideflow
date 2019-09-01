import sys
import os
import slideflow as sf
import argparse
import shutil

from glob import glob

NUM_GPU = 2

sf.set_logging_level(3)
sf.NUM_THREADS = 4
sf.SKIP_VERIFICATION = True

parser = argparse.ArgumentParser(description = "Helper to guide through the SlideFlow pipeline")
parser.add_argument('-q', '--queue', help='Path to queue directory.')
args = parser.parse_args()

if not args.queue or not os.path.exists(args.queue):
	print("You must specify a valid queue directory using the -q flag.")
	sys.exit()

finished_dir = os.path.join(args.queue, "finished")
in_process_dir = os.path.join(args.queue, "in_process")
sys.path.insert(1, in_process_dir)
if not os.path.exists(finished_dir):
    os.makedirs(finished_dir)
if not os.path.exists(in_process_dir):
    os.makedirs(in_process_dir)

while True:
    # Select GPU
    sf.autoselect_gpu(NUM_GPU)
    # Refresh queue
    actions_queue = [py for py in glob(args.queue) if py.split('/')[-1].split('.')[-1] == "py"]
    # Exit if queue empty
    if len(actions_queue) == 0:
        break
    # Get first file and move to in_process_dir
    actions_file = actions_queue[0]
    shutil.move(actions_file, in_process_dir)
    actions_file = os.path.join(in_process_dir, actions_file.split('/')[-1])
    # Import file
    actions_name = actions_file.split('/')[-1].replace('.py', '')
    actions = __import__(actions_name)
    # Create project
    SFP = sf.SlideFlowProject(actions.project)
    # Execute actions
    actions.main(SFP)
    # Move actions file into finished category
    shutil.move(actions_file, finished_dir)
    # Delete old project
    del(SFP)
    # Release GPU
    sf.release_gpu()