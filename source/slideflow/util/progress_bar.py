# Progress bar
# James M Dolezal, (c) 2019

import sys
import time

class ProgressBar:
	def __init__(self, bar_length=20, counter_text=None):
		self.BARS = {}
		self.bar_length = bar_length
		self.next_id = 0
		self.counter = 0
		self.counter_text = "" if not counter_text else " " + counter_text
		self.tail = ''
		self.starttime = None
		self.text = ''

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
		self.hard_refresh()
		return bar_id

	def arrow(self, percent):
		return chr(0x2588) * int(round(percent * self.bar_length))

	def update(self, _id, val, text=None):
		self.BARS[_id].value = min(val, self.BARS[_id].endvalue)
		if text:
			self.BARS[_id].text = text
		self.refresh()

	def refresh(self):
		if not self.starttime:
			self.starttime = time.time()
		if len(self.BARS) == 0:
			sys.stdout.write("\r\033[K")
			sys.stdout.flush()
			return
		out_text = "\r\033[K"
		for i, bar_id in enumerate(self.BARS):
			separator = "  " if i != len(self.BARS)-1 else ""
			bar = self.BARS[bar_id]
			arrow = self.arrow(bar.percent())
			spaces = u'-' * (self.bar_length - len(arrow))
			out_text += u"\u007c{0}\u007c {1}% ({2}){3}".format(arrow + spaces, int(round(bar.percent() * 100)), bar.text, separator)
		out_text += self.tail
		if out_text != self.text:
			sys.stdout.write(out_text)
		self.text = out_text
		sys.stdout.flush()

	def end(self, _id = -1):
		if _id == -1:
			bars_keys = list(self.BARS.keys())
			for bar_id in bars_keys:
				del(self.BARS[bar_id])
				#self.BARS[bar_id].value = self.BARS[bar_id].endvalue
			self.hard_refresh()
			#sys.stdout.write('\n')
		else:
			del(self.BARS[_id])
			self.hard_refresh()

	def print(self, text):
		sys.stdout.write("\r\033[K" + text + "\n")
		sys.stdout.flush()
		self.hard_refresh()

	def hard_refresh(self):
		self.text = ''
		self.refresh()

	def regen_tail(self):
		if not self.starttime:
			self.starttime = time.time()
			return
		self.tail = f" {self.counter/(time.time()-self.starttime):.1f}{self.counter_text}/sec"

	def update_counter(self, value):
		self.counter += value
		self.regen_tail()
		self.refresh()