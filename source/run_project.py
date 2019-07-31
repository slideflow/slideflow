import sys
import slideflow as sf
import argparse

sf.set_logging_level(3)
sf.select_gpu(0)
sf.NUM_THREADS = 4
sf.SKIP_VERIFICATION = True

parser = argparse.ArgumentParser(description = "Helper to guide through the SlideFlow pipeline")
parser.add_argument('-p', '--project', help='Path to project directory.')
args = parser.parse_args()

if not args.project:
	print("You must specify a project directory using the -p flag.")
	sys.exit()

SFP = sf.SlideFlowProject(args.project)

sys.path.insert(0, args.project)
try:
	import actions
except:
	print(f"No actions.py file found in {args.project}")
	sys.exit()

actions.main(SFP)