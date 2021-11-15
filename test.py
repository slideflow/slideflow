import click
import multiprocessing
from slideflow.test import TestSuite

@click.command()
@click.option('--slides', help='Path to directory containing slides', required=True, metavar='DIR')
@click.option('--out', help='Directory in which to store test project files.', required=False, metavar='DIR')
@click.option('--extract', help='Test tile extraction.', required=False, default=False, metavar='<bool>')
@click.option('--reader', help='Test TFRecord readers.', required=False, default=False, metavar='<bool>')
@click.option('--train', help='Test training.', required=False, default=False, metavar='<bool>')
@click.option('--norm', help='Test real-time normalization.', required=False, default=False, metavar='<bool>')
@click.option('--eval', help='Test evaluation.', required=False, default=False, metavar='<bool>')
@click.option('--heatmap', help='Test heatmaps.', required=False, default=False, metavar='<bool>')
@click.option('--mosaic', help='Test mosaic maps.', required=False, default=False, metavar='<bool>')
@click.option('--wsi', help='Test WSI prediction.', required=False, default=False, metavar='<bool>')
@click.option('--clam', help='Test CLAM.', required=False, default=False, metavar='<bool>')
def main(slides, out, extract, reader, train, norm, eval, heatmap, mosaic, wsi, clam):
    if not out:
        out = 'slideflow_test'
    TS = TestSuite(out, slides)
    TS.test(
        extract=extract,
        reader=reader,
        train=train,
        normalizer=norm,
        evaluate=eval,
        heatmap=heatmap,
        activations=mosaic,
        predict_wsi=wsi,
        clam=clam
    )

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main() # pylint: disable=no-value-for-parameter