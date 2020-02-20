import slideflow as sf
import os

if __name__=='__main__':
	root = "/media/ss4tbSSD/TFRecords/"
	project_dirs = [os.path.join(root, d) for d in os.listdir(root)]

	for project_dir in project_dirs:
		sf.util.log.header(f"Updating TFRecords in project {sf.util.green(project_dir)}")
		num_updated = 0
		subfolders = [os.path.join(project_dir, _d) for _d in os.listdir(project_dir)]
		for subfolder in subfolders:
			sf.util.log.empty(f"Updating TFRecords in {sf.util.green(subfolder)}", 1)
			num_updated += sf.io.tfrecords.update_tfrecord_dir(subfolder, slide='case')
		sf.util.log.complete(f"Updated {num_updated} TFRecords.")
