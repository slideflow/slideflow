import csv

with open('/Users/james/results/Thyroid/Test_Set_B/balanced_weights.csv', 'r') as csv_file:
	reader = csv.reader(csv_file, delimiter=',')
	headers = next(reader, None)
	with open('metadata.tsv', 'w') as outfile:
		writer = csv.writer(outfile, delimiter='\t')
		writer.writerow(headers[0:8])
		for row in reader:
			writer.writerow(row[0:8])


