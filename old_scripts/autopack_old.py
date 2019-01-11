import os
from os.path import *
import sys
import re

import packing

SIZE = 128
packing.SIZE = SIZE
case_path = "/home/falafel/histcon/cases/"
train_path = "/home/falafel/histcon/train_data/"
DIR_MAP = {'good':'0', 'normal':'0', 'bad':'1'}
packing.CASE_DIR = case_path
packing.VERBOSE = False
EXPORT=True
r = re.compile(".*json")

try:
    l = os.listdir(case_path)
except FileNotFoundError:
    print("Invalid folder: [%s]" % case_path)

c = list(filter(r.match, os.listdir(case_path)))

if len(c) == 0:
    print("No cases found.")
    sys.exit()

case_list = list(map(lambda x: x[:-5], c))

print("Cases identified:")
for c in case_list: print(" * %s" % c)

for case in case_list:
    print("\nAnalyzing case %s" % case)
    packing.CASE = case
    packing.main(EXPORT, plot=False)
    if EXPORT:
        dirs = [d for d in os.listdir(packing.RESULTS_DIR) if not isfile(join(packing.RESULTS_DIR, d))]
        print()
        for di in dirs:
            if di in DIR_MAP:
                print("Copying %s -> ../%s" % (di, DIR_MAP[di]))
                if not isdir(train_path): os.mkdir(train_path)
                if not isdir(join(train_path, DIR_MAP[di])):
                    os.mkdir(join(train_path, DIR_MAP[di]))
                command = "find %s -name '*.jpg' -exec cp -t %s {} +" % (join(packing.RESULTS_DIR, di), join(train_path, DIR_MAP[di]))
                os.system(command)
        os.system("rm -r %s" % packing.RESULTS_DIR)

