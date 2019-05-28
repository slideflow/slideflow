import slideflow as sf
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = "Helper to guide through the SlideFlow pipeline")
	parser.add_argument('-p', '--project', help='Path to project directory.')
	args = parser.parse_args()

	SFP = sf.SlideFlowProject(args.project)
	SFP.prepare_tiles()