# Progress bar
# James M Dolezal, 2018

import sys

def bar(value, endvalue, bar_length=20, newline=True, text=''):
	percent = float(value) / endvalue
	arrow = chr(0x2588) * int(round(percent * bar_length))# + '>'
	spaces = u' ' * (bar_length - len(arrow))

	if newline:
		sys.stdout.write("\r")
    	
	sys.stdout.write(u"\u007c{0}\u007c {1}% {2}".format(arrow + spaces, int(round(percent * 100)), text))
	sys.stdout.flush()

def end():
	bar(10, 10)
	sys.stdout.flush()
	print('\n')
