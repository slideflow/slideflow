import click
import multiprocessing
from slideflow.test import TestSuite

@click.command()
@click.option('--slides', help='Path to directory containing slides', required=True, metavar='DIR')
@click.option('--out', help='Directory in which to store test project files.', required=False, metavar='DIR')
@click.option('--clam', help='Include CLAM tester.', required=False, default=False, metavar='<bool>')
def main(slides, out, clam):
    if not out:
        out = 'slideflow_test'
    TS = TestSuite(out, slides)
    TS.test(clam=clam)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main() # pylint: disable=no-value-for-parameter