import click
import multiprocessing
from slideflow.test import TestSuite

@click.command()
@click.option('--slides', help='Path to directory containing slides', required=True, metavar='DIR')
@click.option('--out', help='Directory in which to store test project files.', required=False, metavar='DIR')
def main(slides, out):
    if not out:
        out = 'slideflow_test'
    TS = TestSuite(out, slides)
    TS.test()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main() # pylint: disable=no-value-for-parameter