from os.path import join
import tensorflow as tf
import sfutil
import glob

tfrecord_file = "/home/shawarma/data/slideflow_projects/tcga_thca_follicular/tfrecord/train.tfrecords"
tfrecord_folder = "/home/shawarma/data/slideflow_projects/tcga_thca_follicular/tfrecord/train"

FEATURE_DESCRIPTION =  {'category': tf.io.FixedLenFeature([], tf.int64),
						'case':     tf.io.FixedLenFeature([], tf.string),
						'image_raw':tf.io.FixedLenFeature([], tf.string)}

def _float_feature(value):
	"""Returns a bytes_list from a float / double."""
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
	"""Returns a bytes_list from a string / byte."""
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
	"""Returns an int64_list from a bool / enum / int / uint."""
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def image_example(category, case, image_string):
	feature = {
		'category': _int64_feature(category),
		'case':     _bytes_feature(case),
		'image_raw':_bytes_feature(image_string),
	}
	return tf.train.Example(features=tf.train.Features(feature=feature))

def _parse_tfrecord_function(record):
	features = tf.io.parse_single_example(record, FEATURE_DESCRIPTION)
	return features

def split_tfrecord(tfrecord_file):
	dataset = tf.data.TFRecordDataset(tfrecord_file)
	writers = {}
	for record in dataset:
		features = _parse_tfrecord_function(record)
		case = features['case'].numpy()
		category = features['category'].numpy()
		image_raw = features['image_raw'].numpy()
		shortname = sfutil._shortname(case.decode('utf-8'))

		if shortname not in writers.keys():
			tfrecord_path = join(tfrecord_folder, f"{shortname}.tfrecords")
			writer = tf.io.TFRecordWriter(tfrecord_path)
			writers.update({shortname: writer})
		else:
			writer = writers[shortname]
		tf_example = image_example(category, case, image_raw)
		writer.write(tf_example.SerializeToString())

	for case in writers.keys():
		writers[case].close()

def verify_tfrecord(tfrecord_folder):
	# Now, verify the contents
	tfrecord_files = glob.glob(join(tfrecord_folder, "*.tfrecords"))
	for tfr in tfrecord_files:
		v_dataset = tf.data.TFRecordDataset(tfr)
		for record in v_dataset:
			features = _parse_tfrecord_function(record)
			case = features['category'].numpy()
			print(case)

split_tfrecord(tfrecord_file)