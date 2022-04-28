import os
import click
import multiprocessing
import logging
import tabulate  # type: ignore
import slideflow as sf
from slideflow.test import TestSuite
from slideflow.util import colors as col


@click.command()
@click.option('--slides', help='Path to directory containing slides',
              required=False, metavar='DIR')
@click.option('--out', help='Directory in which to store test project files.',
              required=False, metavar='DIR')
@click.option('--all', help='Perform all tests.',
              required=False, type=bool)
@click.option('--extract', help='Test tile extraction.',
              required=False, type=bool)
@click.option('--reader', help='Test TFRecord readers.',
              required=False, type=bool)
@click.option('--train', help='Test training.',
              required=False, type=bool)
@click.option('--norm', 'normalizer', help='Test real-time normalization.',
              required=False, type=bool)
@click.option('--eval', 'evaluate', help='Test evaluation.',
              required=False, type=bool)
@click.option('--predict', help='Test prediction/inference.',
              required=False, type=bool)
@click.option('--heatmap', help='Test heatmaps.',
              required=False, type=bool)
@click.option('--act', 'activations', help='Test activations & mosaic maps.',
              required=False, type=bool)
@click.option('--wsi', 'predict_wsi', help='Test WSI prediction.',
              required=False, type=bool)
@click.option('--clam', help='Test CLAM.',
              required=False, type=bool)
def main(slides, out, all, **kwargs):
    if not out:
        out = 'slideflow_test'
    if 'SF_LOGGING_LEVEL' in os.environ:
        verbosity = logging.getLogger('slideflow').getEffectiveLevel()
    else:
        verbosity = logging.WARNING
    if all is not None:
        kwargs = {k: all if kwargs[k] is None else kwargs[k] for k in kwargs}

    # Show active backend
    if sf.backend() == 'tensorflow':
        print(col.bold("\nActive backend:"), col.yellow('tensorflow'))
    elif sf.backend() == 'torch':
        print(col.bold("\nActive backend:"), col.purple('torch'))
    else:
        print(col.bold("\nActive backend: <Unknown>"))

    # Show tests to run
    print(col.bold("\nTests to run:"))
    tests = kwargs.values()
    print(tabulate.tabulate({
        'Test': kwargs.keys(),
        'Run': [col.green('True') if v else col.red('False') for v in tests]
    }))
    TS = TestSuite(out, slides, verbosity=verbosity)
    TS.test(**kwargs)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()  # pylint: disable=no-value-for-parameter
