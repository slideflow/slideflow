import sys
import slideflow as sf
import argparse

if __name__=='__main__':
	sf.set_logging_level(3)
	sf.autoselect_gpu(2)
	sf.NUM_THREADS = 4
	sf.SKIP_VERIFICATION = False

	parser = argparse.ArgumentParser(description = "Helper to guide through the SlideFlow pipeline")
	parser.add_argument('-p', '--project', help='Path to project directory.')
	args = parser.parse_args()

	if not args.project:
		print("You must specify a project directory using the -p flag.")
		sys.exit()

	SFP = sf.SlideflowProject(args.project)

	sys.path.insert(0, args.project)
	try:
		import actions
	except:
		print(f"Error loading actions.py in {args.project}; either does not exist or contains an error")
		sys.exit()

	actions.main(SFP)