import sys
import slideflow as sf
import argparse
import multiprocessing
import os

if __name__=='__main__':
	multiprocessing.freeze_support()

	parser = argparse.ArgumentParser(description = "Helper to guide through the Slideflow pipeline")
	parser.add_argument('-p', '--project', required=True, help='Path to project directory.')
	parser.add_argument('-g', '--gpu', type=int, help='Manually specify GPU to use.')
	parser.add_argument('-gp', '--gpu_pool', type=int, help='Number of available GPUs in pool, from which to autoselect GPU.')
	
	parser.add_argument('-t', '--threads', type=int, default=4, help='Number of threads to use during tile extraction.')

	parser.add_argument('--nfs', action="store_true", help="Sets environmental variable HDF5_USE_FILE_LOCKING='FALSE' as a fix to problems with NFS file systems.")
	args = parser.parse_args()

	if args.nfs:
		os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
		print("Set environmental variable 'HDF5_USE_FILE_LOCKING'='FALSE'")

	SFP = sf.SlideflowProject(args.project, gpu=args.gpu, gpu_pool=args.gpu_pool)

	SFP.FLAGS['num_threads'] = args.threads

	sys.path.insert(0, args.project)
	try:
		import actions
	except Exception as e:
		print(f"Error loading actions.py in {args.project}; either does not exist or contains an error")
		raise Exception

	actions.main(SFP)
