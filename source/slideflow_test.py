import os
import multiprocessing
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
from slideflow.test import TestSuite

if __name__=='__main__':
	multiprocessing.freeze_support()

	parser = argparse.ArgumentParser(description = "Helper to guide through the Slideflow testing")
	parser.add_argument('-r', '--reset', action="store_true", help='Reset test project.')
	parser.add_argument('-g', '--gpu', type=int, help='Specify GPU for testing..')
	parser.add_argument('-v', '--verbose', action="store_true", help='Reset test project.')
	args = parser.parse_args()

	TS = TestSuite('/media/shawarma/data/sf_test_project', reset=args.reset, gpu=args.gpu, debug=args.verbose)
	TS.test()
