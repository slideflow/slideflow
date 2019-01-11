# Multiprocessing python test
# ===========================

import os
from multiprocessing import Pool

def doubler(num):
	return num * 2
	#_id = os.getpid()
	#print ('{0} doubled to {1} by process id: {2}'.format(num, result, _id))

def double_list(num_list):
	return map(doubler, num_list)

if __name__=='__main__':
	numbers	= [[2, 4, 6, 8, 10], [1, 3, 5, 7, 9]]
	pool = Pool(processes=3)
	print(pool.map(double_list, numbers))