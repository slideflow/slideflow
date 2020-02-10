import sys
import os
import slideflow as sf
import argparse
import shutil
import multiprocessing

from glob import glob

if __name__=='__main__':
	multiprocessing.freeze_support()

	parser = argparse.ArgumentParser(description = "Helper to guide through the SlideFlow pipeline")
	parser.add_argument('-q', '--queue', help='Path to queue directory.')
	parser.add_argument('-g', '--gpu', type=int, default=2, help='Number of available GPUs.')
	parser.add_argument('-t', '--threads', type=int, default=4, help='Number of threads to use during tile extraction.')
	parser.add_argument('-sV', '--skip_verification', action="store_true", help="Whether or not to skip verification.")
	parser.add_argument('-tM', '--test_mode', action="store_true", help="Whether or not to train in test mode.")
	parser.add_argument('--nfs', action="store_true", help="Sets environmental variable HDF5_USE_FILE_LOCKING='FALSE' as a fix to problems with NFS file systems.")
	args = parser.parse_args()

	if args.nfs:
		os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
		print("Set environmental variable 'HDF5_USE_FILE_LOCKING'='FALSE'")
	
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
		# Refresh queue
		actions_queue = [py for py in glob(os.path.join(args.queue, "*")) if sf.util.path_to_ext(py) == "py"]
		# Exit if queue empty
		if len(actions_queue) == 0:
			print("Queue empty; exiting...")
			break
		# Get first file and move to in_process_dir
		actions_file = actions_queue[0]
		print(f"Loading actions file at {sf.sfutil.green(actions_file)}")
		shutil.move(actions_file, in_process_dir)
		actions_file = os.path.join(in_process_dir, actions_file.split('/')[-1])
		# Import file
		actions_name = actions_file.split('/')[-1].replace('.py', '')
		actions = __import__(actions_name)
		# Create project
		SFP = sf.SlideflowProject(actions.project)
		SFP.autoselect_gpu(args.gpu)
		SFP.FLAGS['skip_verification'] = args.skip_verification
		SFP.FLAGS['test_mode'] = args.test_mode
		# Execute actions
		actions.main(SFP)
		# Move actions file into finished category
		shutil.move(actions_file, finished_dir)
    # Release GPU
		SFP.release_gpu()
		# Delete old project
		del(SFP)