import os
import logging
import click
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import slideflow as sf
from tqdm import tqdm
from slideflow.io import tfrecords
from os.path import join, isfile, exists

@click.command()
@click.option('--outdir', help='Where to save filtered tfrecords', required=True, metavar='DIR')
@click.option('--source', help='Source directory for tfrecords to filter', required=True, metavar='DIR')
@click.option('--model', help='Path to PANCAN model', required=True, metavar='DIR')
def main(outdir, source, model):
    k_model = tf.keras.models.load_model(model)
    wsi_tfrecords = [tfr for tfr in os.listdir(source) if isfile(join(source, tfr)) and sf.util.path_to_ext(tfr) == 'tfrecords']

    if not exists(outdir):
        os.makedirs(outdir)

    pb = tqdm(wsi_tfrecords, ncols=80)
    for tfr in pb:
        pb.set_description("Working...")
        num_wrote = 0
        tfr_path = join(source, tfr)

        parser = tfrecords.get_tfrecord_parser(tfr_path, ('image_raw',), decode_images=True, standardize=True, img_size=299)
        pred_dataset = tf.data.TFRecordDataset(tfr_path)
        pred_dataset = pred_dataset.map(parser, num_parallel_calls=8)
        pred_dataset = pred_dataset.batch(128, drop_remainder=False)
        try:
            roi_pred, project_pred = k_model.predict(pred_dataset)
        except ValueError:
            continue

        writer = tf.io.TFRecordWriter(join(outdir, tfr))
        dataset = tf.data.TFRecordDataset(tfr_path)
        parser = tfrecords.get_tfrecord_parser(tfr_path, decode_images=False, to_numpy=True)
        for i, record in enumerate(dataset):
            if roi_pred[i][1] > 0.5:
                writer.write(tfrecords.read_and_return_record(record, parser))
                num_wrote += 1
        tqdm.write(f'Finished {tfr} : wrote {num_wrote}')
        writer.close()

if __name__=='__main__':
    main() # pylint: disable=no-value-for-parameter