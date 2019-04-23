import csv
import argparse
import sys
import umap
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import pickle

import progress_bar

'''csv_started = os.path.exists(join(self.SAVE_FOLDER, 'final_layer_weights.csv'))
write_mode = 'a' if csv_started else 'w'
with open(join(self.SAVE_FOLDER, 'final_layer_weights.csv'), write_mode) as csv_file:
	csv_writer = csv.writer(csv_file, delimiter = ',')
	if not csv_started:
		csv_writer.writerow(["Tile_num", "Case", "Category"] + [f"Logits{l}" for l in range(len(logits[0]))] + [f"Node{n}" for n in range(len(output[0]))])
	for l in range(len(output)):
		out = output[l].tolist()
		logit = logits[l].tolist()
		csv_writer.writerow([labels[l], name, category] + logit + out)'''

def get_args():
	parser = argparse.ArgumentParser(description = 'Creates a t-SNE histology tile mosaic using a saved t-SNE bookmark generated with Tensorboard.')
	parser.add_argument('-f', '--file', help='Path to csv file.')
	parser.add_argument('-m', '--max', type=int, default=10000, help='Maximum number of rows to extract.')
	parser.add_argument('--category', type=int, default=3, help='Column number in which category name is found.')
	parser.add_argument('--case', type=int, default=2, help='Column number in which case name is found.')
	return parser.parse_args()

def normalize_layout(layout, min_percentile=1, max_percentile=99, relative_margin=0.1):
    """Removes outliers and scales layout to between [0,1]."""

    # compute percentiles
    mins = np.percentile(layout, min_percentile, axis=(0))
    maxs = np.percentile(layout, max_percentile, axis=(0))

    # add margins
    mins -= relative_margin * (maxs - mins)
    maxs += relative_margin * (maxs - mins)

    # `clip` broadcasts, `[None]`s added only for readability
    clipped = np.clip(layout, mins, maxs)

    # embed within [0,1] along both axes
    clipped -= clipped.min(axis=0)
    clipped /= clipped.max(axis=0)

    return clipped

def gen_umap(array):
	layout = umap.UMAP(n_components=2, verbose=True, n_neighbors=20, min_dist=0.01, metric="cosine").fit_transform(array)

	## You can optionally use TSNE as well
	# layout = TSNE(n_components=2, verbose=True, metric="cosine", learning_rate=10, perplexity=50).fit_transform(d)

	layout = normalize_layout(layout)

	with open('umap_weights.pkl', 'wb') as handle:
		pickle.dump(layout, handle)

	plt.figure(figsize=(10, 10))
	plt.scatter(x=layout[:,0],y=layout[:,1], s=2)
	plt.show()

def load_weights_umap(path):
	with open(path, 'rb') as handle:
		umap_array = pickle.load(handle)

	print(f"Loaded array of size {len(umap_array)}")
	gen_umap(umap_array)

def start_umap(args):
	csv_path = args.file
	started = False

	umap_array = []

	pb = progress_bar.ProgressBar()
	pb_id = pb.add_bar(0, 114000)
	with open(csv_path, 'r') as file:
		reader = csv.reader(file, delimiter=',')
		headers = next(reader, None)
		for i, row in enumerate(reader):
			pb.update(pb_id, i)
			np_row = np.array([row[8:]]).astype(np.float)
			umap_array += [[float(x) for x in row[8:]]]
			'''if not started:
				umap_array = np_row
				started = True
			else:
				umap_array = np.append(umap_array, np_row, axis=0)'''
	pb.end()

	with open('np_array_weights.pkl', 'wb') as handle:
		pickle.dump(umap_array, handle)

	#print(umap_array.shape)
	gen_umap(umap_array)
	

def main(args):
	categories = {}
	cases = {}
	csv_path = args.file
	with open(csv_path, 'r') as file:
		reader = csv.reader(file, delimiter=',')
		headers = next(reader, None)
		for row in reader:
			case = row[args.case-1]
			category = row[args.category-1]
			if category not in categories:
				categories.update({category:1})
			else:
				categories[category] += 1
			if case not in cases:
				cases.update({case:{'count':1, 'category':category, 'write_count':0}})
			else:
				cases[case]['count'] += 1

	for category in categories:
		print(f"Category {category}: {categories[category]}")

	tiles_per_category = int(args.max / len(list(categories)))
	print(f"Extracting on {tiles_per_category} tiles per category.")

	for category in categories:
		fraction = tiles_per_category / categories[category]
		for case in cases:
			if cases[case]['category'] == category:
				tiles_to_extract = int(fraction * cases[case]['count'])
				print(f"Extracting {tiles_to_extract} of {cases[case]['count']} from {case} [{cases[case]['category']}]")
				keep_list = [1] * tiles_to_extract
				keep_list += [0] * (cases[case]['count'] - tiles_to_extract)
				shuffle(keep_list)
				cases[case]['keep_list'] = keep_list

	with open(csv_path, 'r') as read_file:
		root_dir = '/'.join(csv_path.split('/')[:-1])
		csv_out_path =  join(root_dir, 'balanced.csv')
		print(f"Writing CSV to {csv_out_path}...")
		reader = csv.reader(read_file, delimiter=',')
		headers = next(reader, None)
		with open(csv_out_path, 'w') as write_file:
			writer = csv.writer(write_file, delimiter=',')
			writer.writerow(headers)
			for row in reader:
				case = row[args.case-1]
				row_index = cases[case]['write_count']
				if cases[case]['keep_list'][row_index]:
					writer.writerow(row)
				cases[case]['write_count'] += 1

if __name__ == '__main__':
	args = get_args()
	#main(args)
	start_umap(args)
	#load_weights_umap('np_array_weights.pkl')