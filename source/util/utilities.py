HEADER = '\033[95m'
BLUE = '\033[94m'
GREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

def warn(text):
	return WARNING + text + ENDC

def header(text):
	return HEADER + text + ENDC

def info(text):
	return BLUE + text + ENDC

def green(text):
	return GREEN + text + ENDC

def fail(text):
	return FAIL + text + ENDC

def bold(text):
	return BOLD + text + ENDC

def underline(text):
	return UNDERLINE + text + ENDC