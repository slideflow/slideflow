from slideflow.convoluter import TMAReader
from slideflow.util.progress_bar import ProgressBar

if __name__ == '__main__':
	pb = ProgressBar(bar_length=5, counter_text='tiles')
	sampleTMA = TMAReader("C:\\Users\\James\\Documents\\Research\\TMA\\TMA_1185.svs", "TMA_1185", "svs", 1208, 604, 1, "C:\\Users\\James\\Documents\\Research\\TMA\\tiles", pb=pb)
	sampleTMA.export_tiles(export_full_tma=False)