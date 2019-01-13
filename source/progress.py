import sys

def bar(value, endvalue, bar_length=20, newline=True):
	percent = float(value) / endvalue
	arrow = chr(0x2588) * int(round(percent * bar_length))# + '>'
	spaces = u' ' * (bar_length - len(arrow))

	if newline:
		sys.stdout.write("\r")
    	
	sys.stdout.write(u"\u007c{0}\u007c {1}%".format(arrow + spaces, int(round(percent * 100))))
	sys.stdout.flush()

def end():
	bar(10, 10)
	sys.stdout.flush()
