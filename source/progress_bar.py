# Progress bar
# James M Dolezal, 2018

import sys
import time

class ProgressBar:
	def __init__(self, bar_length=20):
		self.BARS = {}
		self.bar_length = bar_length
		self.next_id = 0

	def add_bar(self, val, endval, endtext=''):
		bar_id = self.next_id
		self.next_id += 1

		class Bar:
			value = val
			endvalue = endval
			text = endtext
			id = bar_id
			def percent(self):
				return float(self.value) / self.endvalue

		self.BARS.update({bar_id: Bar()})
		self.refresh()
		return bar_id

	def arrow(self, percent):
		return chr(0x2588) * int(round(percent * self.bar_length))

	def update(self, _id, val, text=None):
		self.BARS[_id].value = val
		if text:
			self.BARS[_id].text = text
		self.refresh()

	def refresh(self):
		if len(self.BARS) == 0:
			sys.stdout.write("\033[K\r")
		out_text = "\r\033[K"
		for i, bar_id in enumerate(self.BARS):
			separator = "   ||   " if i != len(self.BARS)-1 else ""
			bar = self.BARS[bar_id]
			arrow = self.arrow(bar.percent())
			spaces = u'-' * (self.bar_length - len(arrow))
			out_text += u"\u007c{0}\u007c {1}% ({2}){3}".format(arrow + spaces, int(round(bar.percent() * 100)), bar.text, separator)
		sys.stdout.write(out_text)
		sys.stdout.flush()

	def end(self, _id = -1):
		if _id == -1:
			for bar_id in self.BARS:
				self.BARS[bar_id].value = self.BARS[bar_id].endvalue
			self.refresh()
			sys.stdout.write('\n')
		else:
			del self.BARS[_id]
			self.refresh()

	def print(self, text):
		sys.stdout.write("\r" + text + "\033[K\n")
		sys.stdout.flush()
		self.refresh()

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

if __name__==('__main__'):
	p = ProgressBar()
	id1 = p.add_bar(0,100, "James")
	id2 = p.add_bar(0,500, "Christopher")
	for i in range(300):
		time.sleep(0.1)
		p.update(id1, (i/30)*10)
		if i == 15:
			p.print("Hello there!")
		if i == 30:
			p.print("How are you?")
		else:
			p.update(id2, (i/10)*50)
	p.end()
