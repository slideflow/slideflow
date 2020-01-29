import sys
import slideflow as sf
import argparse
import multiprocessing

if __name__=='__main__':
	multiprocessing.freeze_support()

	parser = argparse.ArgumentParser(description = "Helper to guide through the SlideFlow pipeline")
	parser.add_argument('-p', '--project', required=True, help='Path to project directory.')
	parser.add_argument('-g', '--gpu', type=int, default=2, help='Number of available GPUs.')
	parser.add_argument('-t', '--threads', type=int, default=4, help='Number of threads to use during tile extraction.')
	parser.add_argument('-sV', '--skip_verification', action="store_true", help="Whether or not to skip verification.")
	parser.add_argument('-tM', '--test_mode', action="store_true", help="Whether or not to train in test mode.")
	args = parser.parse_args()

	sf.NUM_THREADS = args.threads

	SFP = sf.SlideflowProject(args.project)
	SFP.autoselect_gpu(args.gpu)
	SFP.FLAGS['skip_verification'] = args.skip_verification
	SFP.FLAGS['test_mode'] = args.test_mode

	sys.path.insert(0, args.project)
	try:
		import actions
	except:
		print(f"Error loading actions.py in {args.project}; either does not exist or contains an error")
		sys.exit()

	actions.main(SFP)